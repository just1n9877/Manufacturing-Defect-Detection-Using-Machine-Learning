import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neu_defect_detection.config import CLASS_CODES
from neu_defect_detection.data import label_from_path, make_stratified_splits, select_split


class DataUtilitiesTest(unittest.TestCase):
    def test_label_from_path_uses_file_prefix(self) -> None:
        self.assertEqual(label_from_path(Path("Cr_1.bmp")), "Cr")
        self.assertEqual(label_from_path(Path("PS_300.bmp")), "PS")

    def test_make_stratified_splits_preserves_class_counts(self) -> None:
        records = []
        for label in CLASS_CODES:
            for index in range(20):
                records.append({"image_path": f"{label}_{index}.bmp", "label": label})

        split_records = make_stratified_splits(records, seed=42, val_size=0.15, test_size=0.15)

        for split, expected_per_class in (("train", 14), ("val", 3), ("test", 3)):
            split_items = select_split(split_records, split)
            counts = {label: 0 for label in CLASS_CODES}
            for item in split_items:
                counts[item["label"]] += 1
            self.assertEqual(set(counts.values()), {expected_per_class})

    def test_select_split_filters_records(self) -> None:
        records = [
            {"image_path": "a.bmp", "label": "Cr", "split": "train"},
            {"image_path": "b.bmp", "label": "Cr", "split": "val"},
            {"image_path": "c.bmp", "label": "Cr", "split": "test"},
        ]

        self.assertEqual([item["image_path"] for item in select_split(records, "val")], ["b.bmp"])


if __name__ == "__main__":
    unittest.main()
