
# scripts/evaluate_research.py

import argparse
import json
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    average_precision_score,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/annotations/test.jsonl")
    parser.add_argument("--threshold", type=float, default=0.55)
    args = parser.parse_args()

    path = Path(args.input)

    if not path.exists():
        raise SystemExit(f"Fichier introuvable : {path}")

    rows = []

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue

        try:
            row = json.loads(line)
            label = int(row["label"])
            score = float(row["score"])

            if label not in (0, 1) or not 0 <= score <= 1:
                raise ValueError("label doit être 0/1 et score entre 0 et 1.")

            rows.append((label, score))

        except Exception as exc:
            raise SystemExit(f"Ligne {line_number} invalide : {exc}")

    if not rows:
        raise SystemExit("Aucune annotation exploitable.")

    y_true = [row[0] for row in rows]
    y_score = [row[1] for row in rows]
    y_pred = [int(score >= args.threshold) for score in y_score]

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    print(f"Nombre de paires : {len(rows)}")
    print(f"Seuil : {args.threshold:.3f}")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(
        "Précision :",
        f"{precision_score(y_true, y_pred, zero_division=0):.4f}",
    )
    print(
        "Rappel :",
        f"{recall_score(y_true, y_pred, zero_division=0):.4f}",
    )
    print(
        "F1 :",
        f"{f1_score(y_true, y_pred, zero_division=0):.4f}",
    )
    print(f"TN={tn} | FP={fp} | FN={fn} | TP={tp}")

    print("\nRapport détaillé :")
    print(classification_report(y_true, y_pred, zero_division=0))

    if len(set(y_true)) == 2:
        print(
            "PR-AUC (Average Precision) :",
            f"{average_precision_score(y_true, y_score):.4f}",
        )
    else:
        print("PR-AUC non calculable : une seule classe dans les annotations.")


if __name__ == "__main__":
    main()
