import os
import argparse

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from model import densenet121, load_state_dict
from my_dataset import MyDataSet
from utils import read_split_data


def set_requires_grad(module: nn.Module, requires_grad: bool):
    for p in module.parameters():
        p.requires_grad_(requires_grad)


def freeze_backbone_only_train_head(model: nn.Module):
    set_requires_grad(model, False)
    set_requires_grad(model.classifier, True)


def unfreeze_last_block(model: nn.Module):
    set_requires_grad(model, False)

    set_requires_grad(model.classifier, True)

    if hasattr(model, "features"):
        if hasattr(model.features, "denseblock4"):
            set_requires_grad(model.features.denseblock4, True)
        if hasattr(model.features, "norm5"):
            set_requires_grad(model.features.norm5, True)


def train_one_epoch(model, optimizer, data_loader, device, epoch, criterion, amp=False, scaler=None):
    model.train()
    running_loss = 0.0
    total_samples = 0

    for step, (images, labels) in enumerate(data_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        bs = labels.size(0)
        running_loss += loss.item() * bs
        total_samples += bs

        if step % 20 == 0:
            lr_show = optimizer.param_groups[0]["lr"]
            print(f"[train][epoch {epoch}] step={step}/{len(data_loader)} loss={loss.item():.4f} lr={lr_show:.6g}")

    return running_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        pred = torch.argmax(logits, dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()

    os.makedirs(args.save_dir, exist_ok=True)

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }

    train_dataset = MyDataSet(train_images_path, train_images_label, data_transform["train"])
    val_dataset = MyDataSet(val_images_path, val_images_label, data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, args.max_workers])
    print(f"Using {nw} dataloader workers every process")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=nw,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=nw,
        collate_fn=val_dataset.collate_fn,
    )

    model = densenet121(num_classes=args.num_classes, memory_efficient=args.memory_efficient).to(device)

    if args.weights:
        if os.path.exists(args.weights):
            load_state_dict(model, args.weights)
        else:
            raise FileNotFoundError(f"not found weights file: {args.weights}")

    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    best_acc = -1.0
    global_epoch = 0

    if args.epochs_head > 0:
        print("\n========== Phase 1: freeze backbone, train classifier head ==========")
        freeze_backbone_only_train_head(model)

        head_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            head_params,
            lr=args.lr_head,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs_head,
            eta_min=args.min_lr,
        )

        for _ in range(args.epochs_head):
            loss = train_one_epoch(
                model, optimizer, train_loader, device,
                epoch=global_epoch, criterion=criterion,
                amp=args.amp, scaler=scaler
            )
            scheduler.step()
            acc = evaluate(model, val_loader, device)

            lr_now = optimizer.param_groups[0]["lr"]
            print(f"[phase1][epoch {global_epoch}] loss={loss:.4f} val_acc={acc:.4f} lr={lr_now:.6g}")

            tb_writer.add_scalar("phase1/loss", loss, global_epoch)
            tb_writer.add_scalar("phase1/val_acc", acc, global_epoch)
            tb_writer.add_scalar("phase1/lr", lr_now, global_epoch)

            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), os.path.join(args.save_dir, "best_v2.pth"))
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"phase1-epoch{global_epoch}_v2.pth"))
            global_epoch += 1

    if args.epochs_ft > 0:
        print("\n========== Phase 2: unfreeze last dense block, fine-tune ==========")
        unfreeze_last_block(model)

        head_pg = list(model.classifier.parameters())
        backbone_pg = []
        if hasattr(model, "features") and hasattr(model.features, "denseblock4"):
            backbone_pg += list(model.features.denseblock4.parameters())
        if hasattr(model, "features") and hasattr(model.features, "norm5"):
            backbone_pg += list(model.features.norm5.parameters())

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_pg, "lr": args.lr_backbone},
                {"params": head_pg, "lr": args.lr_head_ft},
            ],
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs_ft,
            eta_min=args.min_lr,
        )

        for _ in range(args.epochs_ft):
            loss = train_one_epoch(
                model, optimizer, train_loader, device,
                epoch=global_epoch, criterion=criterion,
                amp=args.amp, scaler=scaler
            )
            scheduler.step()
            acc = evaluate(model, val_loader, device)

            lr_b = optimizer.param_groups[0]["lr"]
            lr_h = optimizer.param_groups[1]["lr"]
            print(f"[phase2][epoch {global_epoch}] loss={loss:.4f} val_acc={acc:.4f} lr_backbone={lr_b:.6g} lr_head={lr_h:.6g}")

            tb_writer.add_scalar("phase2/loss", loss, global_epoch)
            tb_writer.add_scalar("phase2/val_acc", acc, global_epoch)
            tb_writer.add_scalar("phase2/lr_backbone", lr_b, global_epoch)
            tb_writer.add_scalar("phase2/lr_head", lr_h, global_epoch)

            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), os.path.join(args.save_dir, "best_v2.pth"))
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"phase2-epoch{global_epoch}_v2.pth"))
            global_epoch += 1

    print(f"\nTraining done. Best val_acc = {best_acc:.4f}. Best weights saved to: {os.path.join(args.save_dir, 'best_v2.pth')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data / device
    parser.add_argument("--data-path", type=str, default="Flowers/train", help="dataset root used by read_split_data")
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save-dir", type=str, default="./weights")

    # batch / workers
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-workers", type=int, default=8)

    # pretrained
    parser.add_argument("--weights", type=str, default="densenet121-a639ec97.pth", help="pretrained weights path (optional)")

    # two-phase schedule
    parser.add_argument("--epochs-head", type=int, default=3, help="Phase1 epochs (head only)")
    parser.add_argument("--epochs-ft", type=int, default=3, help="Phase2 epochs (fine-tune last block)")

    # optimization
    parser.add_argument("--lr-head", type=float, default=1e-3, help="Phase1 head learning rate")
    parser.add_argument("--lr-head-ft", type=float, default=1e-3, help="Phase2 head learning rate")
    parser.add_argument("--lr-backbone", type=float, default=1e-4, help="Phase2 backbone learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--min-lr", type=float, default=1e-6)

    # memory
    parser.add_argument("--amp", action="store_true", help="use mixed precision (CUDA only)")
    parser.add_argument("--memory-efficient", action="store_true", help="use checkpointing inside DenseNet blocks")

    args = parser.parse_args()
    main(args)
