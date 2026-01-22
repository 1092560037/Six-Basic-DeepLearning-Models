import pandas as pd
import matplotlib.pyplot as plt

def plot_alexnet_training_log_xlsx(
    xlsx_path="alexnet_training_log.xlsx",
    sheet_name=0,
    loss_png="loss_curve.png",
    acc_png="acc_curve.png",
):
    """
    Read alexnet_training_log.xlsx and plot:
      1) train_loss curve
      2) val_accuracy curve
    Save to PNG files.
    Excel columns required: epoch, train_loss, val_accuracy
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

    epochs = df["epoch"].astype(int).tolist()
    train_loss = df["train_loss"].astype(float).tolist()
    val_acc = df["val_accuracy"].astype(float).tolist()

    # Loss curve
    plt.figure()
    plt.plot(epochs, train_loss, label="train_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_png, dpi=200)
    plt.close()

    # Accuracy curve (val only)
    plt.figure()
    plt.plot(epochs, val_acc, label="val_accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(acc_png, dpi=200)
    plt.close()

    print(f"Saved: {loss_png}, {acc_png}")


if __name__ == "__main__":
    plot_alexnet_training_log_xlsx(r"D:\Model test\AlexNet\alexnet_training_log_v2.xlsx")

