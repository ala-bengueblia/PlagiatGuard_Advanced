import json
from pathlib import Path

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from .similarity import combined_similarity


def evaluate_pairs(
    path: str,
    threshold: float = 0.55,
) -> dict:

    rows = [
        json.loads(line)
        for line in Path(path).read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    y_true = []
    y_pred = []

    for row in rows:
        result = combined_similarity(
            row["text_a"],
            row["text_b"],
        )

        y_true.append(int(row["label"]))
        y_pred.append(
            int(result["score"] >= threshold)
        )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "n": len(rows),
        "threshold": threshold,
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "false_positive_rate": (
            fp / (fp + tn) if fp + tn else 0.0
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }