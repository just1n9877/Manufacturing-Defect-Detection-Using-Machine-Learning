import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neu_defect_detection.config import DEFAULT_SPLITS_PATH
from neu_defect_detection.data import make_stratified_splits, scan_dataset, write_splits_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create stratified NEU train/val/test splits.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_SPLITS_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = scan_dataset(args.dataset_dir)
    split_records = make_stratified_splits(records, args.seed, args.val_size, args.test_size)
    write_splits_csv(split_records, args.output)

    counts: dict[tuple[str, str], int] = {}
    for record in split_records:
        key = (record["split"], record["label"])
        counts[key] = counts.get(key, 0) + 1

    print(f"Wrote {args.output}")
    for split in ("train", "val", "test"):
        summary = ", ".join(f"{label}:{counts.get((split, label), 0)}" for label in sorted({r["label"] for r in split_records}))
        print(f"{split}: {summary}")


if __name__ == "__main__":
    main()
