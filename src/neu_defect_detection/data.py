import csv
import random
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from .config import CLASS_CODES, CLASS_NAMES, DEFAULT_IMAGE_SIZE


def label_from_path(path: Path) -> str:
    code = path.stem.split("_", 1)[0]
    if code not in CLASS_NAMES:
        raise ValueError(f"Unknown class prefix in {path.name}")
    return code


def scan_dataset(dataset_dir: Path) -> list[dict[str, str]]:
    paths = sorted(dataset_dir.glob("*.bmp"))
    records = [{"image_path": str(path), "label": label_from_path(path)} for path in paths]
    counts = {code: 0 for code in CLASS_CODES}
    for record in records:
        counts[record["label"]] += 1
    missing = [code for code, count in counts.items() if count == 0]
    if missing:
        raise ValueError(f"Missing class images for: {', '.join(missing)}")
    return records


def make_stratified_splits(
    records: list[dict[str, str]],
    seed: int,
    val_size: float,
    test_size: float,
) -> list[dict[str, str]]:
    labels = [record["label"] for record in records]
    train_val, test = train_test_split(
        records,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    train_val_labels = [record["label"] for record in train_val]
    relative_val_size = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val_size,
        random_state=seed,
        stratify=train_val_labels,
    )

    split_records: list[dict[str, str]] = []
    for split_name, split_items in (("train", train), ("val", val), ("test", test)):
        for item in split_items:
            split_records.append(
                {
                    "image_path": item["image_path"],
                    "label": item["label"],
                    "split": split_name,
                }
            )
    return split_records


def write_splits_csv(records: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image_path", "label", "split"])
        writer.writeheader()
        writer.writerows(records)


def load_splits_csv(splits_path: Path) -> list[dict[str, str]]:
    with splits_path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def select_split(records: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    return [record for record in records if record["split"] == split]


def read_gray_resized(image_path: str, image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)


def augment_gray(image: np.ndarray, rng: random.Random) -> np.ndarray:
    augmented = image.copy()
    if rng.random() < 0.5:
        augmented = cv2.flip(augmented, 1)

    angle = rng.uniform(-8.0, 8.0)
    height, width = augmented.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    augmented = cv2.warpAffine(
        augmented,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    alpha = rng.uniform(0.9, 1.1)
    beta = rng.uniform(-10, 10)
    return cv2.convertScaleAbs(augmented, alpha=alpha, beta=beta)
