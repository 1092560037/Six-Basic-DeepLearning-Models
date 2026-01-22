import os
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torchvision import datasets, transforms

from model_v2 import resnet34


def build_classification_report(cm, class_names):
    cm = np.asarray(cm, dtype=np.int64)
    C = cm.shape[0]
    support = cm.sum(axis=1)
    pred_sum = cm.sum(axis=0)
    tp = cm.diagonal()

    precision, recall, f1 = [], [], []
    for i in range(C):
        p = tp[i] / pred_sum[i] if pred_sum[i] > 0 else 0.0
        r = tp[i] / support[i] if support[i] > 0 else 0.0
        f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(f)

    accuracy = tp.sum() / cm.sum() if cm.sum() > 0 else 0.0
    macro_p = sum(precision) / C
    macro_r = sum(recall) / C
    macro_f = sum(f1) / C
    total = support.sum() if support.sum() > 0 else 1
    weighted_p = sum(precision[i] * support[i] for i in range(C)) / total
    weighted_r = sum(recall[i] * support[i] for i in range(C)) / total
    weighted_f = sum(f1[i] * support[i] for i in range(C)) / total

    header = "              precision    recall  f1-score   support"
    lines = [header, ""]
    for i, name in enumerate(class_names):
        lines.append(
            f"{name:>14} {precision[i]:>10.4f} {recall[i]:>9.4f} {f1[i]:>9.4f} {int(support[i]):>9d}"
        )

    lines.append("")
    lines.append(f"{'accuracy':>14} {'':>10} {'':>9} {accuracy:>9.4f} {int(total):>9d}")
    lines.append(f"{'macro avg':>14} {macro_p:>10.4f} {macro_r:>9.4f} {macro_f:>9.4f} {int(total):>9d}")
    lines.append(f"{'weighted avg':>14} {weighted_p:>10.4f} {weighted_r:>9.4f} {weighted_f:>9.4f} {int(total):>9d}")
    return "\n".join(lines)


def try_build_tensorrt(model, fp16: bool = True):
    """
    TensorRT inference acceleration (optional).
    Uses torch_tensorrt if installed; otherwise falls back to PyTorch.
    """
    try:
        import torch_tensorrt
    except Exception:
        return model, False

    enabled_precisions = {torch.float16} if fp16 else {torch.float32}
    # For dynamic batch, torch_tensorrt requires input specs.
    # Here we assume input size (N,3,224,224) with dynamic N.
    trt_model = torch_tensorrt.compile(
        model,
        inputs=[torch_tensorrt.Input((1, 3, 224, 224), min_shape=(1, 3, 224, 224), opt_shape=(8, 3, 224, 224), max_shape=(32, 3, 224, 224))],
        enabled_precisions=enabled_precisions
    )
    return trt_model, True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_tensorrt", action="store_true", help="Use TensorRT (torch_tensorrt) if available")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    # Paths (run in D:\Model test)
    val_dir = r"D:\Model test\Flowers\val"
    weights_path = r".\resNet34.pth"
    class_json_path = r".\class_indices.json"

    assert os.path.isdir(val_dir), f"VAL dir not found: {val_dir}"
    assert os.path.exists(weights_path), f"weights not found: {weights_path}"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform_val)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True
    )
    num_classes = len(val_dataset.classes)

    if os.path.exists(class_json_path):
        with open(class_json_path, "r", encoding="utf-8") as f:
            idx2name = json.load(f)
        class_names = [idx2name[str(i)] for i in range(num_classes)]
    else:
        class_names = val_dataset.classes

    # Model
    model = resnet34(num_classes=num_classes)
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Mixed precision inference (if CUDA)
    criterion = nn.CrossEntropyLoss(reduction="sum")

    # TensorRT acceleration (optional)
    trt_enabled = False
    if args.use_tensorrt and device.type == "cuda":
        model, trt_enabled = try_build_tensorrt(model, fp16=True)
        if trt_enabled:
            print("TensorRT enabled via torch_tensorrt (FP16).")
        else:
            print("TensorRT not available; fallback to PyTorch.")

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    with torch.inference_mode():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(enabled=(device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, labels)

            total_loss += float(loss.item())
            preds = torch.argmax(logits, dim=1)
            total_correct += int((preds == labels).sum().item())
            total_samples += int(labels.size(0))

            for t, p in zip(labels.view(-1), preds.view(-1)):
                cm[int(t), int(p)] += 1

    val_loss = total_loss / max(total_samples, 1)
    val_acc = total_correct / max(total_samples, 1)

    print("\n===== Evaluation on VAL =====")
    print(f"Samples: {total_samples}")
    print(f"Val Loss (CrossEntropy): {val_loss:.4f}")
    print(f"Top-1 Accuracy: {val_acc:.4f}")

    print("\n===== Classification Report (VAL) =====")
    print(build_classification_report(cm.numpy(), class_names))

    print("\n===== Confusion Matrix (rows=true, cols=pred) =====")
    print(cm.numpy())


if __name__ == "__main__":
    main()
