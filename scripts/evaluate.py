import argparse
import json
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)

from plagiatguard.evaluation import evaluate_pairs

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    default="data/annotations.jsonl",
)
parser.add_argument(
    "--threshold",
    type=float,
    default=0.55,
)

args = parser.parse_args()

result = evaluate_pairs(
    args.input,
    args.threshold,
)

print(json.dumps(result, indent=2, ensure_ascii=False))