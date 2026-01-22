import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from net_v2 import MyLeNet5

# 配置
data_root = r"D:\Model test\Flowers\flowers"
seed = 42
batch_size = 64
epochs = 20
lr = 1e-3

device = "cuda" if torch.cuda.is_available() else "cpu"

# 标准化转换
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std =[0.229, 0.224, 0.225]
)

#图像预处理
transform_train = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    normalize,
])

transform_eval = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    normalize,
])

# 两个 dataset,避免 transform 被覆盖
dataset_train_tf = datasets.ImageFolder(root=data_root, transform=transform_train)
dataset_eval_tf  = datasets.ImageFolder(root=data_root, transform=transform_eval)

num_classes = len(dataset_train_tf.classes)
print("classes:", dataset_train_tf.classes)
print("class_to_idx:", dataset_train_tf.class_to_idx)

# 固定划分索引
g = torch.Generator().manual_seed(seed)
n = len(dataset_train_tf)
n_train = int(n * 0.8)
n_val = int(n * 0.1)
n_test = n - n_train - n_val

perm = torch.randperm(n, generator=g).tolist()
train_idx = perm[:n_train]
val_idx   = perm[n_train:n_train + n_val]
test_idx  = perm[n_train + n_val:]

train_ds = Subset(dataset_train_tf, train_idx)
val_ds   = Subset(dataset_eval_tf,  val_idx)
test_ds  = Subset(dataset_eval_tf,  test_idx)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

#模型引入
model = MyLeNet5(num_classes=5, in_channels=3, input_size=64).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def eval_loss_acc(loader):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return total_loss / total, correct / total

# 训练并记录
best_val_acc = 0.0
history = []  # 每行: epoch, train_loss, train_acc, val_loss, val_acc

for epoch in range(1, epochs + 1):
    model.train()
    running_loss, running_correct, running_total = 0.0, 0, 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y.size(0)
        running_correct += (logits.argmax(dim=1) == y).sum().item()
        running_total += y.size(0)

    train_loss = running_loss / running_total
    train_acc  = running_correct / running_total
    val_loss, val_acc = eval_loss_acc(val_loader)

    history.append((epoch, train_loss, train_acc, val_loss, val_acc))
    print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "LeNet5_v2.pth")

#最终测试
model.load_state_dict(torch.load("LeNet5_v2.pth", map_location=device))
test_loss, test_acc = eval_loss_acc(test_loader)
print("Best val_acc:", best_val_acc)
print("Test acc:", test_acc)

# 保存 history 到 CSV
with open("metrics_ReLU.csv", "w", encoding="utf-8") as f:
    f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")
    for row in history:
        f.write(",".join(str(x) for x in row) + "\n")
