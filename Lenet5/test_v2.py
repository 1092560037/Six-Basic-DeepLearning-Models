import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from net_v2 import MyLeNet5

#配置
data_root = r"D:\Model test\Flowers\flowers"
ckpt_path = r"D:\Model test\LeNet5_v2.pth"
seed = 42
batch_size = 64
device = "cuda" if torch.cuda.is_available() else "cpu"

#transform_eval
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std =[0.229, 0.224, 0.225]
)

transform_eval = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    normalize,
])

dataset = datasets.ImageFolder(root=data_root, transform=transform_eval)
classes = dataset.classes
num_classes = len(classes)

print("classes:", classes)
print("class_to_idx:", dataset.class_to_idx)

#复现split
g = torch.Generator().manual_seed(seed)
n = len(dataset)
n_train = int(n * 0.8)
n_val = int(n * 0.1)
n_test = n - n_train - n_val

perm = torch.randperm(n, generator=g).tolist()
test_idx = perm[n_train + n_val:]  
test_ds = Subset(dataset, test_idx)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

#加载模型
model = MyLeNet5(num_classes=5, in_channels=3, input_size=64).to(device)
state = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state)
model.eval()

#测试 + 混淆矩阵
conf = torch.zeros(num_classes, num_classes, dtype=torch.int64)
correct, total = 0, 0

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.size(0)

        for t, p in zip(y.view(-1), pred.view(-1)):
            conf[t.long(), p.long()] += 1

test_acc = correct / total
print("Test acc:", test_acc)

# 保存 confusion matrix 图
cm = conf.cpu().numpy()

plt.figure(figsize=(7, 6))
plt.imshow(cm)
plt.title("Confusion Matrix (Test)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(range(num_classes), classes, rotation=45, ha="right")
plt.yticks(range(num_classes), classes)
plt.colorbar()

# 在每个格子里写上计数
thresh = cm.max() * 0.6  # 用于决定文字颜色（深色格子用白字）
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(
            j, i, int(cm[i, j]),
            ha="center", va="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=10
        )

plt.tight_layout()
plt.savefig(r"D:\Model test\confusion_matrix_v2.png", dpi=200)
plt.close()

print(r"Saved: D:\Model test\confusion_matrix_v2.png")
