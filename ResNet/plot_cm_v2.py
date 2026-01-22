import numpy as np
import matplotlib.pyplot as plt

class_names = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
cm = np.array([
    [45, 18, 0,  2, 11,],
    [35, 43,  1, 18,  8],
    [13,  8,  0,  1, 56],
    [8, 10,  0, 45, 10],
    [10, 12,  2,  4, 70]
], dtype=int)

def plot_cm(cm, class_names, title, out_png, normalize=False):
    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        cm_show = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=(row_sum != 0))
    else:
        cm_show = cm

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_show, interpolation="nearest")
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title=title
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm_show.max() / 2.0 if cm_show.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{cm_show[i, j]:.2f}" if normalize else str(int(cm_show[i, j]))
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if cm_show[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.show()

plot_cm(cm, class_names, "Confusion Matrix_v2 (Counts)", "confusion_matrix_counts.png", normalize=False)
