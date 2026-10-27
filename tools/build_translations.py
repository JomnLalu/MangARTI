"""Step 3: turn sentence pairs into bidirectional Alpaca translation examples.

Every Formal/Manado pair becomes two examples, one for each direction.

Sources:
    --labeled  a labelled JSONL file (from parse_raw.py / dataset_editor.py) where
               each Formal line (`bahasa`) sits next to its Manado line (`manado`)
    --csv      a CSV file whose first two columns are Formal Indonesian, Manado
               (first row is treated as a header). Can be given more than once.

Usage:
    python tools/build_translations.py --labeled data/labeled.jsonl --out data/translations.new.jsonl
    python tools/build_translations.py --labeled data/labeled.jsonl --csv extra_pairs.csv --out out.jsonl

data/translations.jsonl is the published training set. Write to a new file and
compare before replacing it.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prompt import TO_INDONESIAN, TO_MANADO  # noqa: E402


def pairs_from_labeled(path: Path):
    """Yield (formal, manado) from consecutive line pairs in a labelled JSONL file."""
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for i in range(0, len(lines) - 1, 2):
        a, b = lines[i], lines[i + 1]
        by_label = {
            a.get("output", "").strip().lower(): a.get("input", "").strip(),
            b.get("output", "").strip().lower(): b.get("input", "").strip(),
        }
        if "bahasa" in by_label and "manado" in by_label:
            yield by_label["bahasa"], by_label["manado"]
        else:
            print(f"Skipping lines {i + 1} and {i + 2}: not a bahasa/manado pair.", file=sys.stderr)


def pairs_from_csv(path: Path):
    """Yield (formal, manado) from the first two columns of a CSV file."""
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if len(row) >= 2 and row[0].strip() and row[1].strip():
                yield row[0].strip(), row[1].strip()


def to_examples(formal: str, manado: str):
    return [
        {"instruction": TO_INDONESIAN, "input": manado, "output": formal},
        {"instruction": TO_MANADO, "input": formal, "output": manado},
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labeled", type=Path, help="Labelled JSONL file")
    parser.add_argument("--csv", type=Path, action="append", default=[], help="CSV of Formal,Manado pairs")
    parser.add_argument("--out", type=Path, required=True, help="Output JSONL file")
    args = parser.parse_args()

    if not args.labeled and not args.csv:
        parser.error("give at least one source: --labeled and/or --csv")

    pairs = []
    if args.labeled:
        pairs += pairs_from_labeled(args.labeled)
    for csv_path in args.csv:
        pairs += pairs_from_csv(csv_path)

    with args.out.open("w", encoding="utf-8") as f:
        for formal, manado in pairs:
            for example in to_examples(formal, manado):
                f.write(json.dumps(example) + "\n")
    print(f"Wrote {len(pairs) * 2} examples ({len(pairs)} pairs) to {args.out}")


if __name__ == "__main__":
    main()
