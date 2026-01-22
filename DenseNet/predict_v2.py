import os
import json
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np

import matplotlib.pyplot as plt

from model import densenet121
from my_dataset import MyDataSet
from utils import read_split_data

from sklearn.metrics import classification_report, confusion_matrix


@torch.no_grad()
def evaluate_on_val(model, data_loader, device, criterion=None):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)

        if criterion is not None:
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)

        preds = torch.argmax(logits, dim=1)

        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    avg_loss = total_loss / total_samples if criterion is not None else None
    acc = total_correct / total_samples if total_samples > 0 else 0.0

    return avg_loss, acc, all_labels, all_preds


def plot_confusion_matrix(cm, class_names, normalize=False, save_path=None, show=True):
    """
    cm: confusion matrix (ndarray)
    class_names: list[str]
    normalize: whether to normalize each row
    """
    if normalize:
        cm = cm.astype(np.float64)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, where=row_sums != 0)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved confusion matrix to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    val_dataset = MyDataSet(
        images_path=val_images_path,
        images_class=val_images_label,
        transform=data_transform
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=val_dataset.collate_fn
    )

    class_names = None
    if args.class_indices and os.path.exists(args.class_indices):
        with open(args.class_indices, "r", encoding="utf-8") as f:
            class_indict = json.load(f)  # {"0":"daisy", ...}
        class_names = [class_indict[str(i)] for i in range(args.num_classes)]
    else:
        class_names = [str(i) for i in range(args.num_classes)]

    model = densenet121(num_classes=args.num_classes).to(device)
    assert os.path.exists(args.weights), f"weights not found: {args.weights}"
    model.load_state_dict(torch.load(args.weights, map_location=device))

    criterion = nn.CrossEntropyLoss()

    val_loss, val_acc, y_true, y_pred = evaluate_on_val(model, val_loader, device, criterion)

    print("\n========== Evaluation on VAL_v2 ==========")
    print(f"VAL samples: {len(val_dataset)}")
    print(f"VAL loss   : {val_loss:.4f}")
    print(f"VAL acc    : {val_acc:.4f}")

    print("\n========== Classification Report (VAL)_v2 ==========")
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
    print(report)

    print("========== Confusion Matrix (VAL)_v2 ==========")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    if args.plot_cm:
        plot_confusion_matrix(
            cm,
            class_names=class_names,
            normalize=args.normalize_cm,
            save_path=args.cm_save_path if args.cm_save_path else None,
            show=not args.no_show
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="Flowers/val", help="dataset root used by read_split_data")
    parser.add_argument("--weights", type=str, default="weights/best_v2.pth", help="trained model weights")
    parser.add_argument("--class-indices", type=str, default="class_indices.json", help="class index json")
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--plot-cm", action="store_true", help="plot confusion matrix")
    parser.add_argument("--normalize-cm", action="store_true", help="normalize confusion matrix")
    parser.add_argument("--cm-save-path", type=str, default="", help="path to save confusion matrix figure")
    parser.add_argument("--no-show", action="store_true", help="do not show matplotlib window")

    args = parser.parse_args()
    main(args)
