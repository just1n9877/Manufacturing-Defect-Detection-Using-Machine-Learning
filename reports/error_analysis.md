# Error Analysis and Production Notes

This report summarizes the completed HOG + SVM and ResNet18 experiments on the NEU steel surface defect test split.

## Observed Misclassification Patterns

The test set contains 270 images, with 45 images per class.

### HOG + SVM

Based on the HOG + SVM confusion matrix, the baseline struggled mainly with texture-similar classes such as `Pa`/Patches and `PS`/Pitted surface. HOG + SVM reached 61.48% accuracy, 60.98% macro precision, and 61.48% macro recall.

- `Pa` recall: 18/45. The largest error was `Pa -> Cr` with 12 samples.
- `PS` recall: 15/45. The largest errors were `PS -> Cr` with 11 samples, `PS -> In` with 8 samples, and `PS -> Pa` with 7 samples.
- `RS` was easiest for the baseline, with 45/45 correct predictions.

This pattern is expected for a HOG baseline. HOG captures local edge orientation and gradient histograms, but it does not learn higher-level surface texture. Broad patch-like defects, pitted regions, and crack-like structures can share similar local gradients after grayscale resizing.

### ResNet18

ResNet18 Transfer Learning reached 99.63% accuracy, 99.64% macro precision, and 99.63% macro recall. Based on the ResNet18 confusion matrix, the model made one test-set error:

- Image: `In_240.bmp`
- True class: `In`/Inclusion
- Predicted class: `PS`/Pitted surface
- Saved preview: `reports/misclassified_samples/resnet18_001_true_In_pred_PS.png`

The likely reason is that inclusion and pitted surface samples can both contain small, dark, local defect patterns, and the difference may depend on local density and distribution. ResNet18 learned the broader texture differences well enough that this ambiguity appeared only once on the test set.

## Why ResNet18 Improved the Baseline

ResNet18 can combine local edges with wider spatial context and multi-scale texture cues. That matters for separating patches, pitted surfaces, inclusions, scratches, and crazing, where local gradient features alone are not always distinctive.

## Reducing False Negatives in Production

False negatives are missed defects, so a production system should prioritize recall over raw accuracy.

Recommended controls:

- Tune decision thresholds for high defect recall, especially for safety-critical classes.
- Add a low-confidence review bucket instead of forcing every image into one class.
- Monitor per-class recall and false negative rate, not only overall accuracy.
- Collect more borderline and hard-negative samples from the real line.
- Use active learning: send uncertain or frequently confused samples to human review, then retrain.
- Calibrate cameras, lighting, exposure, and surface cleaning so acquisition drift does not hide defects.
- Consider a two-stage system: first defect detection or segmentation, then defect classification.
- Add periodic validation on recent production images to catch distribution shift early.
