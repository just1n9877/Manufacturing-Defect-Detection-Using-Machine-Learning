from pathlib import Path


CLASS_NAMES = {
    "Cr": "Crazing",
    "In": "Inclusion",
    "Pa": "Patches",
    "PS": "Pitted surface",
    "RS": "Rolled-in scale",
    "Sc": "Scratches",
}

CLASS_CODES = list(CLASS_NAMES.keys())
DEFAULT_IMAGE_SIZE = (200, 200)
DEFAULT_SPLITS_PATH = Path("data/splits.csv")
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_MODEL_DIR = Path("models")
