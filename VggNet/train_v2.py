import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm

import torchvision.models as models
from torchvision.models import VGG16_Weights


def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    # 使用 ImageNet 预训练权重
    weights = VGG16_Weights.IMAGENET1K_V1
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])
    }

    # 数据路径
    data_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    image_path = os.path.join(data_root, "Flowers")
    assert os.path.exists(image_path), f"{image_path} path does not exist."

    train_dataset = datasets.ImageFolder(
        root=os.path.join(image_path, "train"),
        transform=data_transform["train"]
    )
    train_num = len(train_dataset)

    # 保存类别索引
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    with open("class_indices.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(cla_dict, indent=4, ensure_ascii=False))

    # 适当调小 batch，并开启 pin_memory
    batch_size = 16  # 减小批量大小
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    pin_memory = torch.cuda.is_available()

    print(f"Using {nw} dataloader workers every process")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=pin_memory,
        persistent_workers=(nw > 0),
    )

    validate_dataset = datasets.ImageFolder(
        root=os.path.join(image_path, "val"),
        transform=data_transform["val"]
    )
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(
        validate_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin_memory,
        persistent_workers=(nw > 0),
    )

    print(f"using {train_num} images for training, {val_num} images for validation.")

    # 加载 vgg16 预训练 + 替换最后分类层为 5 类
    num_classes = 5
    net = models.vgg16(weights=weights)

    # 替换 classifier 最后一层
    in_features = net.classifier[6].in_features
    net.classifier[6] = nn.Linear(in_features, num_classes)

    net.to(device)

    # Phase 1：冻结卷积特征层，只训练分类头
    set_requires_grad(net.features, False)  # 冻结卷积特征提取部分
    set_requires_grad(net.classifier, True)  # 训练分类头

    loss_function = nn.CrossEntropyLoss()

    # 只优化需要梯度的参数
    head_params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.AdamW(head_params, lr=1e-3, weight_decay=1e-4)

    # 学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # AMP 混合精度，进一步省显存、提速
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # 训练轮数
    epochs_stageA = 6

    # Phase 2：解冻最后一段卷积微调
    do_finetune = True
    epochs_stageB = 4
    finetune_backbone_lr = 1e-4  # backbone 用更小学习率

    best_acc = 0.0
    save_path = "./vgg16_transfer_best.pth"

    def train_one_epoch(epoch_idx: int, total_epochs: int):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for images, labels in train_bar:
            optimizer.zero_grad(set_to_none=True)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = net(images)
                loss = loss_function(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            train_bar.desc = f"train epoch[{epoch_idx+1}/{total_epochs}] loss:{loss.item():.3f}"

        return running_loss / max(1, len(train_loader))

    @torch.no_grad()
    def evaluate():
        net.eval()
        acc = 0.0
        val_bar = tqdm(validate_loader, file=sys.stdout)
        for val_images, val_labels in val_bar:
            val_images = val_images.to(device, non_blocking=True)
            val_labels = val_labels.to(device, non_blocking=True)
            outputs = net(val_images)
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, val_labels).sum().item()
        return acc / val_num

    total_epochs = epochs_stageA + (epochs_stageB if do_finetune else 0)
    current_epoch = 0

    for _ in range(epochs_stageA):
        train_loss = train_one_epoch(current_epoch, total_epochs)
        val_acc = evaluate()
        scheduler.step()

        print(f"[epoch {current_epoch+1}] train_loss: {train_loss:.3f}  val_accuracy: {val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), save_path)
        current_epoch += 1

    if do_finetune:
        # 解冻 features 的最后若干层
        # VGG16 features 是 Sequential，索引越大越靠后
        for idx, layer in enumerate(net.features):
            if idx >= 24:
                set_requires_grad(layer, True)

        # 重新构建 param groups，使得分类头学习率大一些，backbone 学习率小一些
        backbone_params = [p for p in net.features.parameters() if p.requires_grad]
        head_params = [p for p in net.classifier.parameters() if p.requires_grad]

        optimizer = optim.AdamW(
            [
                {"params": backbone_params, "lr": finetune_backbone_lr},
                {"params": head_params, "lr": 5e-4},
            ],
            weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_stageB)

        for _ in range(epochs_stageB):
            train_loss = train_one_epoch(current_epoch, total_epochs)
            val_acc = evaluate()
            scheduler.step()

            print(f"[epoch {current_epoch+1}] train_loss: {train_loss:.3f}  val_accuracy: {val_acc:.3f}")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(net.state_dict(), save_path)
            current_epoch += 1

    print(f"Finished Training. Best val_accuracy: {best_acc:.3f}")
    print(f"Saved best weights to: {save_path}")


if __name__ == "__main__":
    main()
