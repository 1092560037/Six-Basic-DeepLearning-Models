import csv
import matplotlib.pyplot as plt

epochs = []
train_loss, train_acc = [], []
val_loss, val_acc = [], []

with open("metrics_v2.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        epochs.append(int(row["epoch"]))
        train_loss.append(float(row["train_loss"]))
        train_acc.append(float(row["train_acc"]))
        val_loss.append(float(row["val_loss"]))
        val_acc.append(float(row["val_acc"]))

# Loss curve
plt.figure()
plt.plot(epochs, train_loss, label="train_loss")
plt.plot(epochs, val_loss, label="val_loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve_v2.png", dpi=200)
plt.close()

# Accuracy curve
plt.figure()
plt.plot(epochs, train_acc, label="train_acc")
plt.plot(epochs, val_acc, label="val_acc")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("acc_curve_v2.png", dpi=200)
plt.close()

print("Saved: loss_curve_v2.png, acc_curve_v2.png")
