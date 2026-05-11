# Error Analysis and Production Notes

This report should be updated after running both models and inspecting:

- `outputs/hog_svm/confusion_matrix.png`
- `outputs/resnet18/confusion_matrix.png`
- `outputs/*/classification_report.txt`

## Likely Confusions to Check

`Cr` vs `Sc`: crazing and scratches both produce thin, high-contrast line patterns. HOG may confuse them because both classes have strong local edge responses. A CNN should improve this by learning whether the line pattern is irregular and network-like (`Cr`) or more directional and isolated (`Sc`).

`RS` vs `Pa`: rolled-in scale and patches can both appear as wider textured regions. If the defect boundary is weak after resizing, the classifier may rely on local texture density instead of global shape.

`In` vs `PS`: inclusion and pitted surface defects can both appear as small dark spots. The difference is often in spot density, depth impression, and distribution, which can be weakened by grayscale normalization.

## Why HOG + SVM May Fail

HOG captures edge orientation and local gradient histograms. It is useful as a baseline, but it does not learn higher-level context. It is expected to struggle when two defect classes have similar edge direction, scale, or contrast.

## Why ResNet18 Should Help

ResNet18 can learn multi-scale texture and shape cues after transfer learning. It can combine local edge information with wider spatial context, which is important for separating scratches, crazing, pitted surfaces, and broad patches.

## Reducing False Negatives in Production

False negatives are missed defects, so the production goal should prioritize recall over raw accuracy.

Recommended controls:

- Tune decision thresholds for high defect recall, especially for safety-critical classes.
- Add a low-confidence review bucket instead of forcing every image into one class.
- Monitor per-class recall and false negative rate, not only overall accuracy.
- Collect more borderline and hard-negative samples from the real line.
- Use active learning: send uncertain or frequently confused samples to human review, then retrain.
- Calibrate cameras, lighting, exposure, and surface cleaning so acquisition drift does not hide defects.
- Consider a two-stage system: first defect detection/segmentation, then defect classification.
- Add periodic validation on recent production images to catch distribution shift early.
