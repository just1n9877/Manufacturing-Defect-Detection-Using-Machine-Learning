# Manufacturing Defect Detection Using Machine Learning

NEU steel surface defect classification project for a resume-ready portfolio. It includes a classical computer vision baseline and a transfer-learning CNN.

## Apple MDE Relevance

Apple MDE teams need reliable inspection signals to catch surface defects before they become yield loss, rework, or customer-facing quality escapes. This project shows how a manufacturing defect classifier can compare a classical vision baseline against a ResNet18 transfer-learning model, then use confusion matrices and misclassified samples to understand inspection risk.

## Task

Classify six steel surface defects from the NEU dataset:

| Code | Class |
| --- | --- |
| `Cr` | Crazing |
| `In` | Inclusion |
| `Pa` | Patches |
| `PS` | Pitted surface |
| `RS` | Rolled-in scale |
| `Sc` | Scratches |

## Approach

1. OpenCV preprocessing: grayscale loading, resize to `200x200`, and training-only augmentation.
2. Baseline: HOG features + linear SVM.
3. Deep model: PyTorch ResNet18 transfer learning.
4. Evaluation: accuracy, macro precision, macro recall, classification report, confusion matrix.
5. Error analysis: use the confusion matrix and misclassified samples to explain which defect classes are confused and why.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data Split

The original dataset can stay outside the repo. This command creates `data/splits.csv` with stratified train/val/test splits:

```powershell
python scripts/prepare_dataset.py --dataset-dir "C:\Users\18747\Desktop\NEU surface defect database"
```

Default split is 70% train, 15% validation, 15% test.

Note: `data/splits.csv` should be regenerated on each machine because image paths depend on the local dataset location. The split file is ignored by Git; keep only `data/README.md` in the repository.

## Reproducibility Settings

- Random seed: 42
- Image size: 200x200
- Split: 70% train, 15% validation, 15% test
- HOG + SVM: HOG features with LinearSVC and balanced class weights
- ResNet18: ImageNet pretrained weights, AdamW optimizer, learning rate 3e-4, batch size 32, epochs 15
- Model selection: best validation accuracy

## Train Baseline

```powershell
python scripts/train_hog_svm.py --splits data/splits.csv
```

Outputs:

- `outputs/hog_svm/metrics.json`
- `outputs/hog_svm/classification_report.txt`
- `outputs/hog_svm/confusion_matrix.png`
- `models/hog_svm.joblib`

## Train ResNet18

```powershell
python scripts/train_resnet18.py --splits data/splits.csv --epochs 15 --batch-size 32
```

To save the full training log:

```powershell
python scripts/train_resnet18.py --splits data/splits.csv --epochs 15 --batch-size 32 *> reports/training_log_resnet18.txt
```

The first pretrained run may download ImageNet weights. If the machine has no network access, use:

```powershell
python scripts/train_resnet18.py --splits data/splits.csv --epochs 15 --batch-size 32 --no-pretrained
```

Outputs:

- `outputs/resnet18/metrics.json`
- `outputs/resnet18/classification_report.txt`
- `outputs/resnet18/confusion_matrix.png`
- `models/resnet18.pth`

## Experiment Results

Test set size: 270 images, with 45 images per class.

| Model | Accuracy | Macro Precision | Macro Recall |
|---|---:|---:|---:|
| HOG + SVM | 61.48% | 60.98% | 61.48% |
| ResNet18 Transfer Learning | 99.63% | 99.64% | 99.63% |

The ResNet18 transfer-learning model improved test accuracy from 61.48% to 99.63% compared with the HOG + SVM baseline.

Result artifacts committed under `reports/`:

- `reports/metrics_hog_svm.json`
- `reports/metrics_resnet18.json`
- `reports/classification_report_hog_svm.txt`
- `reports/classification_report_resnet18.txt`
- `reports/confusion_hog_svm.json`
- `reports/confusion_resnet18.json`
- `reports/figures/hog_svm_confusion_matrix.png`
- `reports/figures/resnet18_confusion_matrix.png`
- `reports/misclassified_samples/resnet18_001_true_In_pred_PS.png`

## Error Analysis

Based on the HOG + SVM confusion matrix, the baseline performed weakly on `Pa`/Patches and `PS`/Pitted surface. The largest confusion pairs were `Pa -> Cr` with 12 samples, `PS -> Cr` with 11 samples, and `PS -> In` with 8 samples. This is consistent with HOG relying on local edge and gradient patterns instead of higher-level defect texture.

Based on the ResNet18 confusion matrix, the model made one test-set error: `In_240.bmp`, true class `In`/Inclusion, predicted as `PS`/Pitted surface. A likely reason is that both classes contain small dark local defect patterns, and the difference may depend on local density and distribution.

See `reports/error_analysis.md` for the detailed analysis.

## Production-Oriented Inspection Workflow

1. Camera captures the steel surface image.
2. Preprocessing normalizes size, grayscale format, and contrast.
3. The CNN classifier predicts defect class and confidence score.
4. Low-confidence or high-risk predictions enter manual review.
5. Per-class recall and false negative rate are monitored weekly.
6. Misclassified and borderline samples are added to the retraining set.

## Limitations

- The NEU dataset is relatively small and clean compared with real production images.
- The test set contains 270 images, so 99.63% accuracy corresponds to one misclassified image.
- Real production deployment would require validation on recent line images, camera and lighting drift checks, and a recall-oriented threshold.
- This project is a classification prototype, not a full production inspection system.

In a real production line, reducing false negatives matters more than maximizing overall accuracy. Practical measures include using a defect/non-defect threshold tuned for high recall, reviewing low-confidence predictions, collecting more hard negative and borderline defect samples, monitoring per-class recall, and adding lighting/camera checks so missed defects are not caused by image acquisition drift.
