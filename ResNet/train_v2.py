import os
import sys
import json
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms, datasets
from tqdm import tqdm

from model_v2 import resnet34


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def init_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank


def get_rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def main():
    if is_distributed():
        local_rank = init_distributed()
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        local_rank = 0

    if is_main_process():
        print(f"using {device} device.")
        if is_distributed():
            print(f"Distributed: WORLD_SIZE={dist.get_world_size()}  RANK={dist.get_rank()}  LOCAL_RANK={local_rank}")


    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    data_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))  # D:\Model test
    image_path = os.path.join(data_root, "Flowers")  # D:\Model test\Flowers
    assert os.path.exists(image_path), f"{image_path} path does not exist."

    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    val_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                       transform=data_transform["val"])

    train_num = len(train_dataset)
    val_num = len(val_dataset)

    if is_main_process():
        flower_list = train_dataset.class_to_idx
        cla_dict = dict((val, key) for key, val in flower_list.items())
        with open("class_indices.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(cla_dict, indent=4, ensure_ascii=False))

    batch_size = 16

    if is_distributed():
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    nw = 0
    if is_main_process():
        print(f"Using {nw} dataloader workers every process")
        print(f"using {train_num} images for training, {val_num} images for validation.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                              num_workers=nw, sampler=train_sampler, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=nw, sampler=val_sampler, pin_memory=True)

    net = resnet34(num_classes=1000)

    # load ImageNet pretrained weights (local file)
    model_weight_path = "./resnet34.pth"
    assert os.path.exists(model_weight_path), f"file {model_weight_path} does not exist."

    # load checkpoint/state_dict robustly, handle common formats and mismatched fc
    ckpt = torch.load(model_weight_path, map_location="cpu", weights_only=False)

    # unwrap checkpoint if it contains a 'state_dict' entry
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # remove 'module.' prefix from keys produced by DataParallel/DistributedDataParallel
    cleaned_state = {}
    for k, v in state_dict.items():
        new_k = k.replace("module.", "") if k.startswith("module.") else k
        cleaned_state[new_k] = v

    # if checkpoint's fc layer shape doesn't match current model, skip fc params
    fc_w_key = "fc.weight"
    fc_b_key = "fc.bias"
    if fc_w_key in cleaned_state:
        if cleaned_state[fc_w_key].shape != net.fc.weight.shape:
            print(f"Checkpoint fc shape {cleaned_state[fc_w_key].shape} != model {net.fc.weight.shape}; skipping fc from checkpoint")
            cleaned_state.pop(fc_w_key, None)
            cleaned_state.pop(fc_b_key, None)

    net.load_state_dict(cleaned_state, strict=False)

    # replace fc for 5 classes
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 5)
    net.to(device)

    if is_distributed():
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank)

    loss_function = nn.CrossEntropyLoss()

    # Optimizer + LR schedule (ExponentialDecay with staircase)
    # (match the strategy shown in screenshot)
    initial_lr = 0.1
    decay_steps = 100000
    decay_rate = 0.96

    optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=0.9, nesterov=True)

    # staircase decay: lr = initial_lr * decay_rate ** (global_step // decay_steps)
    def lr_lambda(global_step: int):
        k = global_step // decay_steps
        return decay_rate ** k

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Mixed precision
    scaler = GradScaler(enabled=torch.cuda.is_available())

    epochs = 3
    save_path = "./resNet34.pth"
    best_acc = 0.0
    global_step = 0

    for epoch in range(epochs):
        if is_distributed():
            train_sampler.set_epoch(epoch)

        net.train()
        running_loss = 0.0

        if is_main_process():
            train_bar = tqdm(train_loader, file=sys.stdout)
        else:
            train_bar = train_loader

        for step, (images, labels) in enumerate(train_bar):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=torch.cuda.is_available()):
                logits = net(images)
                loss = loss_function(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # step LR scheduler per-iteration (like TensorFlow decay_steps)
            scheduler.step()
            global_step += 1

            running_loss += loss.item()

            if is_main_process():
                train_bar.desc = f"train epoch[{epoch + 1}/{epochs}] loss:{loss.item():.3f} lr:{optimizer.param_groups[0]['lr']:.6f}"

        # Validate (no grad, AMP)
        net.eval()
        correct = 0
        with torch.no_grad():
            if is_main_process():
                val_bar = tqdm(val_loader, file=sys.stdout)
            else:
                val_bar = val_loader

            for val_images, val_labels in val_bar:
                val_images = val_images.to(device, non_blocking=True)
                val_labels = val_labels.to(device, non_blocking=True)

                with autocast(enabled=torch.cuda.is_available()):
                    outputs = net(val_images)

                predict_y = torch.max(outputs, dim=1)[1]
                correct += torch.eq(predict_y, val_labels).sum().item()

                if is_main_process():
                    val_bar.desc = f"valid epoch[{epoch + 1}/{epochs}]"

        # gather accuracy across processes
        if is_distributed():
            correct_tensor = torch.tensor(correct, device=device, dtype=torch.long)
            dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
            correct = int(correct_tensor.item())

        val_acc = correct / val_num

        if is_main_process():
            train_steps = len(train_loader)
            print(f"[epoch {epoch + 1}] train_loss: {running_loss / train_steps:.3f}  val_accuracy: {val_acc:.3f}")

            if val_acc > best_acc:
                best_acc = val_acc
                # save underlying module when using DDP
                to_save = net.module.state_dict() if hasattr(net, "module") else net.state_dict()
                torch.save(to_save, save_path)

    if is_main_process():
        print("Finished Training")

    if is_distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
