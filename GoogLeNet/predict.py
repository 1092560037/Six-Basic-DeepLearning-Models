import os
import json

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

from model import GoogLeNet

from sklearn.metrics import classification_report, confusion_matrix


def load_class_indices(json_path: str):
    """
    class_indices.json: {"0":"daisy","1":"dandelion",...}
    返回：
      idx_to_class: dict[int,str]
      class_names: list[str] (按 idx 升序)
    """
    assert os.path.exists(json_path), f"file: '{json_path}' does not exist."
    with open(json_path, "r", encoding="utf-8") as f:
        idx_to_class_raw = json.load(f)

    idx_to_class = {int(k): v for k, v in idx_to_class_raw.items()}
    class_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
    return idx_to_class, class_names


def get_data_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


@torch.no_grad()
def evaluate_on_val(model, device, val_root, transform, class_names, batch_size=32, num_workers=4):
    """
    在 val 集上输出：
    - accuracy
    - average loss
    - classification report
    - confusion matrix
    """
    assert os.path.exists(val_root), f"VAL path not found: {val_root}"

    val_dataset = datasets.ImageFolder(root=val_root, transform=transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    criterion = nn.CrossEntropyLoss()

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_num = 0

    y_true = []
    y_pred = []

    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)  # [B, num_classes]
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)

        total_correct += (preds == labels).sum().item()
        total_num += images.size(0)

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    avg_loss = total_loss / max(total_num, 1)
    acc = total_correct / max(total_num, 1)

    print("\n========== Evaluation on VAL ==========")
    print(f"VAL samples: {total_num}")
    print(f"VAL loss   : {avg_loss:.4f}")
    print(f"VAL acc    : {acc:.4f}")

    print("\n========== Classification Report (VAL) ==========")
    print(classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4
    ))

    print("\n========== Confusion Matrix (VAL) ==========")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    plt.imshow(cm)  # 不指定颜色，使用 matplotlib 默认
    plt.title("Confusion Matrix (VAL)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.show()


@torch.no_grad()
def predict_single_image(model, device, img_path, transform, idx_to_class):
    assert os.path.exists(img_path), f"file: '{img_path}' does not exist."
    img = Image.open(img_path).convert("RGB")

    plt.imshow(img)

    x = transform(img)
    x = torch.unsqueeze(x, dim=0).to(device)  # [1, C, H, W]

    model.eval()
    output = model(x).squeeze(0).cpu()  # [num_classes]
    prob = torch.softmax(output, dim=0)
    pred_idx = int(torch.argmax(prob).item())

    print_res = f"class: {idx_to_class[pred_idx]}   prob: {prob[pred_idx].item():.3f}"
    plt.title(print_res)

    for i in range(len(prob)):
        print(f"class: {idx_to_class[i]:10}   prob: {prob[i].item():.3f}")

    plt.show()


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = get_data_transform()

    json_path = "./class_indices.json"
    idx_to_class, class_names = load_class_indices(json_path)

    model = GoogLeNet(num_classes=5, aux_logits=False).to(device)

    weights_path = "./googleNet.pth"
    assert os.path.exists(weights_path), f"file: '{weights_path}' does not exist."
    model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)

    val_root = r"D:\Model test\Flowers\val"

    evaluate_on_val(
        model=model,
        device=device,
        val_root=val_root,
        transform=transform,
        class_names=class_names,
        batch_size=32,
        num_workers=4
    )

    # 单张图预测
    img_path = "tulip.jpg"
    predict_single_image(
        model=model,
        device=device,
        img_path=img_path,
        transform=transform,
        idx_to_class=idx_to_class
    )



if __name__ == "__main__":
    main()
