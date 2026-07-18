# GD-Net
### 面向微控制器 (MCU) 的轻量化目标检测框架 (Lightweight Object Detection Framework for MCUs)

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![TFLite](https://img.shields.io/badge/TFLite-Micro-orange?style=for-the-badge&logo=tensorflow)](https://www.tensorflow.org/lite/microcontrollers)
[![ONNX](https://img.shields.io/badge/ONNX-Deployable-005CED?style=for-the-badge&logo=onnx)](https://onnx.ai/)
[![Params](https://img.shields.io/badge/Params-1.016M-brightgreen?style=for-the-badge)](#)
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)](LICENSE)

</div>

> 本项目是一个针对微控制器 (MCU) 等资源受限设备设计的轻量化目标检测框架。集成了 ProxylessNAS/MCUNet 主干网络与 YOLO 风格检测流程，支持 ONNX / TFLite 端侧部署。

---

## 项目简介

- **轻量化检测模型**：参数量约 1.016 M，支持 INT8 量化，适配 SRAM 与 Flash 资源受限的硬件。
- **现代检测架构**：YOLOv3-PANet + SPPF + 解耦头 (Decoupled Head)，多尺度融合，加速收敛。
- **NAS 驱动主干网**：支持 MCUNet-VWW / MCUNet-ImageNet / LSNet 等多种配置，覆盖不同资源预算。
- **端侧部署**：内置 ONNX 与 TFLite 导出脚本，适配 TensorFlow Lite for Microcontrollers 等嵌入式推理后端。
- **多格式数据**：兼容 PASCAL VOC (XML) 和 YOLO (TXT) 两种标注格式。

---

## 检测演示
<div align="center">
  <img src="assets/zidane.jpg" alt="检测结果展示" width="80%">
</div>

---

## 系统架构

| 模块 | 功能 | 关键文件 | 可选配置 |
| :--- | :--- | :--- | :--- |
| **Backbone** | 特征提取主干网络 | `gd_net/backbone_mcunet.py`, `backbone_lsnet.py` | `mcunet-vww` / `mcunet-imagenet` / `lsnet-t` |
| **Neck** | 感受野增强 (SPPF) | `gd_net/neck.py` | 池化核大小可调 |
| **FPN** | 多尺度特征融合 (PANet) | `gd_net/fpn.py` | `yolov3_panet` |
| **Head** | 解耦检测头 | `gd_net/head.py` | 分类/回归分支分离 |
| **Assigner** | Anchor 匹配策略 | `gd_net/assigner.py` | IoU 阈值可配 |
| **Loss** | 多任务损失 | `gd_net/loss.py` | 分类/回归/目标性权重可调 |

> 所有超参数集中在 `cfg.py` 管理：Anchor 尺寸、损失权重、骨干网络选择、检测头配置等。

---

## 性能指标 (@320×320 输入)

| 指标 | 数值 | 备注 |
| :--- | :--- | :--- |
| **参数量 (Params)** | 1.016 M | 远小于主流边缘模型 |
| **计算量 (FLOPs)** | 651.09 M | 适配 Cortex-M 系列 |
| **Flash 占用** | 4.90 MB (Float32) | INT8 量化后可进一步缩小 |

---

## 主干网络配置

| 模型 ID | 分辨率 | 适用场景 |
| :--- | :---: | :--- |
| `mcunet-vww` | 160×160 | 精度与速度平衡，适用于大多数 MCU 场景 |
| `mcunet-imagenet` | 224×224 | 更高计算资源的边缘设备 |
| `lsnet-t` | 224×224 | 侧重高帧率检测 |

> 预训练权重加载路径硬编码在 `gd_net/backbone_mcunet.py` 中，使用前请检查路径。

---

## 快速上手

### 1. 环境依赖
- **Python**: 3.8+
- **Core**: PyTorch 1.10+, Torchvision
- **Tools**: OpenCV, PyYAML, thop
- **TFLite 导出 (可选)**: onnx2tf

### 2. 安装
```bash
git clone https://github.com/your-repo/GD_Net.git
cd GD_Net
pip install torch torchvision opencv-python PyYAML thop onnx onnxruntime
```

### 3. 数据准备

支持 VOC (XML) 和 YOLO (TXT) 两种格式，通过 `.yaml` 中 `label_format` 字段区分：

```yaml
# VOC 格式: path/Annotations/*.xml, path/ImageSets/Main/{train,val}.txt, path/JPEGImages/*.jpg
path: /path/to/dataset_root
label_format: voc
train: ImageSets/Main/train.txt
val: ImageSets/Main/val.txt
names: {0: drone}

# YOLO 格式: path/images/{train,val}/*.jpg + path/labels/{train,val}/*.txt
path: /path/to/dataset_root
label_format: yolo
train: images/train
val: images/val
names: {0: car}
```

### 4. 主要用法
```bash
# 模型训练
python train.py --data data/vehicle.yaml --img-size 320 --batch-size 64 --epochs 300

# 模型推理
python predict.py \
  --source assets/zidane.jpg \
  --weights checkpoints/car_yolov3_mcu.pth \
  --data data/standford.yaml \
  --output det_results

# 模型评估
python val.py --data data/vehicle.yaml --weights checkpoints/best_yolov3_mcu.pth       # YOLO 格式
python val_voc.py --data data/drone.yaml --weights checkpoints/best_yolov3_mcu.pth     # VOC 格式

# 模型导出
python export.py --weights checkpoints/best_yolov3_mcu.pth --data data/vehicle.yaml    # ONNX
python export_tflite.py --weights checkpoints/best_yolov3_mcu.pth --data data/vehicle.yaml  # TFLite

# 端侧推理
python predict_onnx.py --source assets/img.png --onnx checkpoints/car_mcu.onnx --data data/vehicle.yaml
python predict_tflite.py --source assets/img.png --tflite checkpoints/car_mcu.tflite --data data/vehicle.yaml
```

---

> **项目维护者注**：如需更多关于模型架构或训练策略的细节，请参考 `gd_net/` 下各模块源码或联系开发团队。
