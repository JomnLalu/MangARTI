"""Step 1: parse the raw text files into a labelled JSONL file.

Each line in data/raw/*.txt is `<sentence> /<label>`, where the label is `bahasa`
(Formal Indonesian) or `manado`. Lines alternate: a Formal sentence, then its
Manado translation.

Usage:
    python tools/parse_raw.py                    # writes data/labeled.raw.jsonl
    python tools/parse_raw.py --out data/labeled.jsonl --overwrite

data/labeled.jsonl has been hand-corrected with tools/dataset_editor.py, so this
script will not overwrite it unless you pass --overwrite.
"""

import argparse
import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INSTRUCTION = "Classify the language form as either Dialect or Formal."


def parse_file(path: Path):
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        if "/" not in line:
            print(f"Skipping {path.name}:{line_no} (no label): {line!r}", file=sys.stderr)
            continue
        sentence, label = line.rsplit("/", 1)
        yield {"instruction": INSTRUCTION, "input": sentence.strip(), "output": label.strip().lower()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=DATA_DIR / "raw")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "labeled.raw.jsonl")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing output file")
    args = parser.parse_args()

    if args.out.exists() and not args.overwrite:
        sys.exit(f"{args.out} already exists. Pass --overwrite to replace it.")

    files = sorted(args.raw_dir.glob("*.txt"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    rows = [row for path in files for row in parse_file(path)]

    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} rows from {len(files)} files to {args.out}")


if __name__ == "__main__":
    main()
