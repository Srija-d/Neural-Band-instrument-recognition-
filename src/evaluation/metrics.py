"""Shared evaluation so every model is scored the same way. Owner: Srija."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score)

from src.config import CLASS_CODES, RESULTS_DIR


def evaluate(y_true, y_pred, model_name, run_name="seed0", save=True):
    """Return macro-F1 and accuracy; optionally save metrics JSON and confusion matrix."""
    scores = {
        "model": model_name,
        "run": run_name,
        "macro_f1": f1_score(y_true, y_pred, labels=CLASS_CODES, average="macro"),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    if save:
        out = Path(RESULTS_DIR) / model_name
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{run_name}_metrics.json").write_text(json.dumps(scores, indent=2))
        cm = confusion_matrix(y_true, y_pred, labels=CLASS_CODES)
        disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_CODES)
        fig, ax = plt.subplots(figsize=(7, 6))
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{model_name} ({run_name})  macro-F1 = {scores['macro_f1']:.3f}")
        fig.tight_layout()
        fig.savefig(out / f"{run_name}_confusion.png", dpi=150)
        plt.close(fig)
    return scores


def majority_baseline(y_train, y_test):
    """'Always guess the most common instrument' baseline."""
    most_common = max(set(y_train), key=list(y_train).count)
    return [most_common] * len(y_test)


# TODO (week 6): plot macro-F1 vs. training fraction for all models (data-scarcity curves).
