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
    roc_auc_score,
    roc_curve,
    average_precision_score,
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="data/annotations/test.jsonl",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
    )

    args = parser.parse_args()

    path = Path(args.input)

    if not path.exists():
        raise SystemExit(
            f"Fichier introuvable : {path}"
        )

    rows = []

    for line_number, line in enumerate(
        path.read_text(
            encoding="utf-8"
        ).splitlines(),
        start=1,
    ):

        if not line.strip():
            continue

        try:
            row = json.loads(line)

            label = int(row["label"])
            score = float(row["score"])

            if label not in (0, 1):
                raise ValueError(
                    "label doit être 0 ou 1."
                )

            if not 0 <= score <= 1:
                raise ValueError(
                    "score doit être compris entre 0 et 1."
                )

            rows.append({
                "label": label,
                "score": score,
                "language": row.get("language", "unknown"),
                "text_type": row.get("text_type", "original"),
            })

        except Exception as exc:
            raise SystemExit(
                f"Ligne {line_number} invalide : {exc}"
            )

    if not rows:
        raise SystemExit(
            "Aucune annotation exploitable."
        )

    y_true = [row["label"] for row in rows]
    y_score = [row["score"] for row in rows]

    y_pred = [
        int(score >= args.threshold)
        for score in y_score
    ]

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    print(f"Nombre de paires : {len(rows)}")
    print(f"Seuil : {args.threshold:.3f}")

    print(
        "Accuracy :",
        f"{accuracy_score(y_true, y_pred):.4f}",
    )

    print(
        "Precision :",
        f"{precision_score(y_true, y_pred, zero_division=0):.4f}",
    )

    print(
        "Recall :",
        f"{recall_score(y_true, y_pred, zero_division=0):.4f}",
    )

    print(
        "F1-score :",
        f"{f1_score(y_true, y_pred, zero_division=0):.4f}",
    )

    print(
        f"TN={tn} | FP={fp} | FN={fn} | TP={tp}"
    )

    print("\nRapport détaillé :")

    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    if len(set(y_true)) == 2:

        print(
            "ROC-AUC :",
            f"{roc_auc_score(y_true, y_score):.4f}",
        )

        print(
            "PR-AUC :",
            f"{average_precision_score(y_true, y_score):.4f}",
        )

        fpr, tpr, thresholds = roc_curve(
            y_true,
            y_score,
        )

        roc_data = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
        }

        Path(
            "reports"
        ).mkdir(exist_ok=True)

        Path(
            "reports/roc_curve.json"
        ).write_text(
            json.dumps(
                roc_data,
                indent=2,
            ),
            encoding="utf-8",
        )

    else:
        print(
            "ROC-AUC non calculable : "
            "une seule classe dans les annotations."
        )

    print("\nÉvaluation par type de texte :")

    categories = sorted({
        row["text_type"]
        for row in rows
    })

    for category in categories:

        indices = [
            i
            for i, row in enumerate(rows)
            if row["text_type"] == category
        ]

        true_subset = [
            y_true[i]
            for i in indices
        ]

        pred_subset = [
            y_pred[i]
            for i in indices
        ]

        print(f"\nType : {category}")
        print(f"Nombre de paires : {len(indices)}")

        print(
            "Accuracy :",
            f"{accuracy_score(true_subset, pred_subset):.4f}",
        )

        print(
            "F1-score :",
            f"{f1_score(true_subset, pred_subset, zero_division=0):.4f}",
        )

    print("\nÉvaluation par langue :")

    languages = sorted({
        row["language"]
        for row in rows
    })

    for language in languages:

        indices = [
            i
            for i, row in enumerate(rows)
            if row["language"] == language
        ]

        true_subset = [
            y_true[i]
            for i in indices
        ]

        pred_subset = [
            y_pred[i]
            for i in indices
        ]

        print(f"\nLangue : {language}")
        print(f"Nombre de paires : {len(indices)}")

        print(
            "Accuracy :",
            f"{accuracy_score(true_subset, pred_subset):.4f}",
        )

        print(
            "F1-score :",
            f"{f1_score(true_subset, pred_subset, zero_division=0):.4f}",
        )


if __name__ == "__main__":
    main()