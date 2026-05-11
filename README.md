# Manufacturing Defect Detection Using Machine Learning

NEU steel surface defect classification project for a resume-ready portfolio. It includes a classical computer vision baseline and a transfer-learning CNN.

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
| --- | ---: | ---: | ---: |
| HOG + SVM | 61.48% | 60.98% | 61.48% |
| ResNet18 transfer learning | 99.63% | 99.64% | 99.63% |

The ResNet18 transfer-learning model improved test accuracy from 61.48% to 99.63% compared with the HOG + SVM baseline.

## Resume Bullets

- Built a steel surface defect classifier on the NEU dataset covering six industrial defect categories.
- Implemented an OpenCV preprocessing and augmentation pipeline, plus a HOG + SVM baseline for interpretable classical CV comparison.
- Fine-tuned a ResNet18 transfer-learning model in PyTorch and evaluated accuracy, macro precision/recall, and confusion matrices.
- Analyzed class-level misclassification patterns and proposed production controls to reduce false negatives.

中文简历版本：

- 基于 NEU 钢材表面缺陷数据集完成 6 类缺陷分类，覆盖 crazing、inclusion、patches、pitted surface、rolled-in scale、scratches。
- 使用 OpenCV 实现灰度化、resize 和训练集数据增强，构建 HOG + SVM 传统视觉 baseline。
- 使用 PyTorch fine-tune ResNet18 迁移学习模型，并输出 accuracy、macro precision、macro recall 和 confusion matrix。
- 基于混淆矩阵分析易混类别，并提出真实产线中降低 false negative 的策略。

## Error Analysis Guide

After training, inspect `outputs/*/classification_report.txt` and `outputs/*/confusion_matrix.png`.

Typical NEU confusion patterns to check:

- In this experiment, HOG + SVM performed weakly on `Pa`/Patches and `PS`/Pitted surface, while ResNet18 made only very few mistakes, mainly likely between small dark-spot defects such as `In`/Inclusion and `PS`/Pitted surface.
- `RS` vs `Pa`: both can form broad texture regions; boundaries may be weak after grayscale resize.
- `Cr` vs `Sc`: both have thin line-like structures; cracks are often irregular, scratches more directional.
- `In` vs `PS`: both can appear as small dark local defects; scale and density matter.

In a real production line, reducing false negatives matters more than maximizing overall accuracy. Practical measures include using a defect/non-defect threshold tuned for high recall, reviewing low-confidence predictions, collecting more hard negative and borderline defect samples, monitoring per-class recall, and adding lighting/camera checks so missed defects are not caused by image acquisition drift.
