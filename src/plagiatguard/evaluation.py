import json
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
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

    if not rows:
        raise ValueError(
            "Le fichier d'annotations est vide."
        )

    y_true = []
    y_score = []

    for row in rows:

        result = combined_similarity(
            row["text_a"],
            row["text_b"],
        )

        label = int(row["label"])

        if label not in (0, 1):
            raise ValueError(
                "Les labels doivent être 0 ou 1."
            )

        score = float(result["score"])

        if not 0 <= score <= 1:
            raise ValueError(
                "Les scores doivent être compris entre 0 et 1."
            )

        y_true.append(label)
        y_score.append(score)

    y_pred = [
        int(score >= threshold)
        for score in y_score
    ]

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    result = {
        "n": len(rows),
        "threshold": threshold,

        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),

        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },

        "roc_auc": None,
        "roc_curve": None,
    }

    if len(set(y_true)) == 2:

        result["roc_auc"] = float(
            roc_auc_score(
                y_true,
                y_score,
            )
        )

        fpr, tpr, thresholds = roc_curve(
            y_true,
            y_score,
        )

        result["roc_curve"] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
        }

    return result