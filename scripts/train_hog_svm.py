import argparse
from pathlib import Path
import random
import sys

import joblib
import cv2
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neu_defect_detection.config import DEFAULT_MODEL_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_SPLITS_PATH
from neu_defect_detection.data import augment_gray, load_splits_csv, read_gray_resized, select_split
from neu_defect_detection.metrics import save_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HOG + SVM baseline on NEU defects.")
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "hog_svm")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_DIR / "hog_svm.joblib")
    parser.add_argument("--augmentations", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_hog() -> cv2.HOGDescriptor:
    return cv2.HOGDescriptor(
        _winSize=(200, 200),
        _blockSize=(40, 40),
        _blockStride=(20, 20),
        _cellSize=(20, 20),
        _nbins=9,
    )


def hog_feature(hog: cv2.HOGDescriptor, image: np.ndarray) -> np.ndarray:
    return hog.compute(image).reshape(-1)


def build_features(records: list[dict[str, str]], augmentations: int, seed: int) -> tuple[np.ndarray, list[str]]:
    hog = make_hog()
    rng = random.Random(seed)
    features: list[np.ndarray] = []
    labels: list[str] = []
    for record in tqdm(records, desc="HOG train images"):
        image = read_gray_resized(record["image_path"])
        images = [image]
        for _ in range(augmentations):
            images.append(augment_gray(image, rng))
        for item in images:
            features.append(hog_feature(hog, item))
            labels.append(record["label"])
    return np.asarray(features), labels


def build_eval_features(records: list[dict[str, str]]) -> tuple[np.ndarray, list[str]]:
    hog = make_hog()
    features: list[np.ndarray] = []
    labels: list[str] = []
    for record in tqdm(records, desc="HOG test images"):
        features.append(hog_feature(hog, read_gray_resized(record["image_path"])))
        labels.append(record["label"])
    return np.asarray(features), labels


def main() -> None:
    args = parse_args()
    records = load_splits_csv(args.splits)
    train_records = select_split(records, "train")
    test_records = select_split(records, "test")

    print(f"Train images: {len(train_records)}, test images: {len(test_records)}", flush=True)
    x_train, y_train = build_features(train_records, args.augmentations, args.seed)
    x_test, y_test = build_eval_features(test_records)
    print(f"Training features: {x_train.shape}, test features: {x_test.shape}", flush=True)

    model = make_pipeline(
        StandardScaler(),
        LinearSVC(C=1.0, class_weight="balanced", dual="auto", max_iter=args.max_iter, random_state=args.seed),
    )
    print("Training LinearSVC...", flush=True)
    model.fit(x_train, y_train)
    print("Evaluating...", flush=True)
    predictions = model.predict(x_test).tolist()

    metrics = save_evaluation(y_test, predictions, args.output_dir)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_path)
    print(metrics)


if __name__ == "__main__":
    main()
