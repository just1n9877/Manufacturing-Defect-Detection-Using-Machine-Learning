# 项目总结：NEU 钢材表面缺陷分类

## 项目目标

基于 NEU steel surface defect 数据集，构建一个可展示在简历中的制造业缺陷分类项目。项目覆盖传统机器学习 baseline 和深度学习迁移学习方案，并输出可解释的评估结果。

## 技术路线

数据处理：

- 使用 OpenCV 读取 `.bmp` 图片。
- 将图片转为灰度图并 resize 到 `200x200`。
- 只对训练集做数据增强，包括水平翻转、小角度旋转、亮度和对比度扰动。

模型：

- HOG + Linear SVM：作为传统计算机视觉 baseline。
- ResNet18 transfer learning：使用 ImageNet 预训练模型 fine-tune 到 6 类缺陷分类。

评估：

- Accuracy
- Macro precision
- Macro recall
- Classification report
- Confusion matrix

## 误判分析方向

重点关注以下类别混淆：

- `Cr` 和 `Sc`：都可能呈现细长线状纹理，HOG 特征容易只捕捉到相似边缘方向。
- `RS` 和 `Pa`：都可能表现为较大面积纹理区域，resize 后边界信息可能变弱。
- `In` 和 `PS`：都可能呈现小暗点，主要差异在分布密度和局部形态。

## 真实产线降低 False Negative 的策略

False negative 表示缺陷被漏检，真实产线中通常比 false positive 更危险。建议：

- 将模型阈值调向高召回率，而不是只追求最高 accuracy。
- 对低置信度样本进入人工复核队列。
- 按类别监控 recall 和 false negative rate。
- 持续收集真实产线中的边界样本、难样本和新批次样本。
- 定期检查光照、相机、曝光和表面清洁状态，降低图像采集漂移。
- 对高风险场景使用两阶段方案：先做缺陷定位/分割，再做分类。
