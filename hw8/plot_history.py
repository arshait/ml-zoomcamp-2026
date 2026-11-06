"""Plot baseline and augmented histories. Run after reference_train.py."""
from pathlib import Path

import matplotlib.pyplot as plt
import json

root = Path(__file__).parent
baseline = json.loads((root / "history_baseline.json").read_text(encoding="utf-8"))
augmented = json.loads((root / "history_augmented.json").read_text(encoding="utf-8"))
plots = root / "plots"
plots.mkdir(exist_ok=True)

def save(name, series_a, series_b, ylabel):
    plt.figure(figsize=(8, 4.5))
    epochs_a = range(1, len(series_a) + 1)
    epochs_b = range(1, len(series_b) + 1)
    plt.plot(epochs_a, series_a, marker="o", label="baseline")
    plt.plot(epochs_b, series_b, marker="o", label="augmented")
    plt.xlabel("Epoch within phase")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots / name, dpi=140)
    plt.close()

save("loss_curves.png", baseline["train_loss"], augmented["train_loss"], "train loss")
save("eval_loss_curves.png", baseline["evaluation_loss"], augmented["evaluation_loss"], "evaluation loss")
save("accuracy_curves.png", baseline["train_accuracy"], augmented["train_accuracy"], "train accuracy")
save("eval_accuracy_curves.png", baseline["evaluation_accuracy"], augmented["evaluation_accuracy"], "evaluation accuracy")
print("wrote plots")
