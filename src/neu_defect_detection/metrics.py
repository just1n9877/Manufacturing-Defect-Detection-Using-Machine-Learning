import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)

from .config import CLASS_CODES, CLASS_NAMES


def save_evaluation(y_true: list[str], y_pred: list[str], output_dir: Path) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=CLASS_CODES, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=CLASS_CODES, average="macro", zero_division=0)),
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    report = classification_report(
        y_true,
        y_pred,
        labels=CLASS_CODES,
        target_names=[CLASS_NAMES[code] for code in CLASS_CODES],
        zero_division=0,
    )
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    save_confusion_matrix(y_true, y_pred, output_dir / "confusion_matrix.png")
    return metrics


def save_confusion_matrix(y_true: list[str], y_pred: list[str], output_path: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_CODES)
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(CLASS_CODES)), labels=CLASS_CODES)
    ax.set_yticks(np.arange(len(CLASS_CODES)), labels=CLASS_CODES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] > threshold else "black"
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
