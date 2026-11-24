import csv
import io
import json


def to_json(report: dict) -> str:
    return json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
    )


def to_csv(report: dict) -> str:
    output = io.StringIO()

    fields = [
        "source",
        "score",
        "lexical_score",
        "semantic_score",
    ]

    writer = csv.DictWriter(
        output,
        fieldnames=fields,
    )
    writer.writeheader()

    for match in report.get("matches", []):
        writer.writerow({
            key: match.get(key, "")
            for key in fields
        })

    return output.getvalue()