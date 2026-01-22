import os
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.models import VGG16_Weights
import torchvision.models as models

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt


def build_model(num_classes=5, device="cpu"):
    model = models.vgg16()
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    model.to(device)
    return model


@torch.no_grad()
def evaluate(
    data_dir,
    weights_path="vgg16_transfer_best.pth",
    batch_size=32,
    num_workers=4,
    topk=(1, 3),
    device=None,
    save_cm_path=None
):
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    weights = VGG16_Weights.DEFAULT
    transform = weights.transforms()

    val_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    num_classes = len(val_dataset.classes)
    model = build_model(num_classes=num_classes, device=device)

    assert os.path.exists(weights_path), f"weights not found: {weights_path}"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    criterion = nn.CrossEntropyLoss(reduction="sum")

    y_true = []
    y_pred = []
    y_prob = []  # softmax 概率，用于 top-k / 置信度分析
    total_loss = 0.0
    total = 0

    # top-k correct 计数
    correct_at_k = {k: 0 for k in topk}

    for images, labels in val_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item()

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        # top-k 统计
        for k in topk:
            topk_idx = torch.topk(probs, k=k, dim=1).indices  # [B, k]
            correct_at_k[k] += (topk_idx == labels.unsqueeze(1)).any(dim=1).sum().item()

        y_true.append(labels.cpu().numpy())
        y_pred.append(preds.cpu().numpy())
        y_prob.append(probs.cpu().numpy())

        total += labels.size(0)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_prob = np.concatenate(y_prob)

    # 指标
    acc_top1 = correct_at_k.get(1, (y_true == y_pred).mean() * total) / total
    acc_topk = {k: correct_at_k[k] / total for k in topk}
    val_loss = total_loss / total

    # 混淆矩阵与分类报告
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    report = classification_report(
        y_true, y_pred,
        target_names=val_dataset.classes,
        digits=4
    )

    # 输出
    print("===== Evaluation on VAL =====")
    print(f"Samples: {total}")
    print(f"Val Loss (CrossEntropy): {val_loss:.4f}")
    for k in topk:
        print(f"Top-{k} Accuracy: {acc_topk[k]:.4f}")

    print("\n===== Classification Report (P/R/F1) =====")
    print(report)

    print("===== Confusion Matrix =====")
    print(cm)

    if save_cm_path:
        plt.figure(figsize=(7, 6))
        plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, val_dataset.classes, rotation=45, ha="right")
        plt.yticks(tick_marks, val_dataset.classes)

        thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(j, i, format(cm[i, j], "d"),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.savefig(save_cm_path, dpi=200)
        print(f"Saved confusion matrix figure to: {save_cm_path}")

    return {
        "val_loss": val_loss,
        "acc_topk": acc_topk,
        "cm": cm,
        "classes": val_dataset.classes
    }


if __name__ == "__main__":
    data_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    val_dir = os.path.join(data_root, "Flowers", "val")

    evaluate(
        data_dir=val_dir,
        weights_path="vgg16_transfer_best.pth",
        batch_size=32,
        num_workers=4,
        topk=(1, 3),
        save_cm_path="confusion_matrix.png"
    )
