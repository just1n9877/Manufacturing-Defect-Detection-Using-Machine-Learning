# Results Summary

Dataset: NEU steel surface defect classification.

Split: 70% train, 15% validation, 15% test. The test set contains 270 images, with 45 images per class.

| Model | Accuracy | Macro Precision | Macro Recall |
|---|---:|---:|---:|
| HOG + SVM | 61.48% | 60.98% | 61.48% |
| ResNet18 Transfer Learning | 99.63% | 99.64% | 99.63% |

## Evidence Files

- `reports/metrics_hog_svm.json`
- `reports/metrics_resnet18.json`
- `reports/classification_report_hog_svm.txt`
- `reports/classification_report_resnet18.txt`
- `reports/confusion_hog_svm.json`
- `reports/confusion_resnet18.json`
- `reports/figures/hog_svm_confusion_matrix.png`
- `reports/figures/resnet18_confusion_matrix.png`
- `reports/misclassified_samples/resnet18_001_true_In_pred_PS.png`

## Main Takeaway

The classical HOG + SVM baseline provides a useful comparison point but struggles with texture classes such as Patches and Pitted surface. ResNet18 transfer learning performs much better on this clean benchmark test set, with one misclassified Inclusion sample predicted as Pitted surface.
