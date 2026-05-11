import argparse
import copy
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neu_defect_detection.config import CLASS_CODES, DEFAULT_MODEL_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_SPLITS_PATH
from neu_defect_detection.data import augment_gray, load_splits_csv, read_gray_resized, select_split
from neu_defect_detection.metrics import save_evaluation


LABEL_TO_INDEX = {label: index for index, label in enumerate(CLASS_CODES)}
INDEX_TO_LABEL = {index: label for label, index in LABEL_TO_INDEX.items()}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class NEUDataset(Dataset):
    def __init__(self, records: list[dict[str, str]], train: bool):
        self.records = records
        self.train = train

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        image = read_gray_resized(record["image_path"])
        if self.train:
            image = augment_gray(image, random)

        image = image.astype(np.float32) / 255.0
        image = np.stack([image, image, image], axis=0)
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        label = LABEL_TO_INDEX[record["label"]]
        return torch.from_numpy(image), torch.tensor(label, dtype=torch.long)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune ResNet18 on NEU defects.")
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "resnet18")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_DIR / "resnet18.pth")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(pretrained: bool) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_CODES))
    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[str], list[str]]:
    model.eval()
    y_true: list[str] = []
    y_pred: list[str] = []
    for images, labels in tqdm(loader, desc="eval", leave=False):
        images = images.to(device)
        logits = model(images)
        predictions = logits.argmax(dim=1).cpu().tolist()
        y_pred.extend(INDEX_TO_LABEL[index] for index in predictions)
        y_true.extend(INDEX_TO_LABEL[index] for index in labels.tolist())
    return y_true, y_pred


def accuracy_from_labels(y_true: list[str], y_pred: list[str]) -> float:
    correct = sum(true == pred for true, pred in zip(y_true, y_pred))
    return correct / max(1, len(y_true))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records = load_splits_csv(args.splits)
    train_records = select_split(records, "train")
    val_records = select_split(records, "val")
    test_records = select_split(records, "test")

    train_loader = DataLoader(
        NEUDataset(train_records, train=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        NEUDataset(val_records, train=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        NEUDataset(test_records, train=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_accuracy = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_true, val_pred = predict(model, val_loader, device)
        val_accuracy = accuracy_from_labels(val_true, val_pred)
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_accuracy={val_accuracy:.4f}")
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = {
                "model_state": copy.deepcopy(model.state_dict()),
                "class_codes": CLASS_CODES,
                "val_accuracy": best_val_accuracy,
            }

    if best_state is not None:
        model.load_state_dict(best_state["model_state"])
        args.model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, args.model_path)

    test_true, test_pred = predict(model, test_loader, device)
    metrics = save_evaluation(test_true, test_pred, args.output_dir)
    print(metrics)


if __name__ == "__main__":
    main()
