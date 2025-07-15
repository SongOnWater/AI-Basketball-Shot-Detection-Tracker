# 实时篮球投篮检测（YOLOv8与OpenCV结合使用）
作者：[Avi Shah](https://www.linkedin.com/in/-avishah/) （2023年）

得分检测准确率：95% <br>
投篮检测准确率：97% <br>

https://github.com/avishah3/AI-Basketball-Shot-Detector-Tracker/assets/115107522/469750e1-0f3c-4b07-9fc5-5387831742f7

## 语言选择
- [English](README.md)
- [中文](README-zh.md)

## 项目简介

本项目结合了机器学习和计算机视觉的强大功能，旨在实时检测和分析篮球投篮！基于最新的YOLOv8（You Only Look Once）机器学习模型和OpenCV库，程序可以处理来自各种来源的视频流，如实时摄像头或预录制的视频，为用户提供一种身临其境的游戏体验，并增强比赛分析能力。

## 目录结构

```
├── README.md
├── best.pt             # 训练好的模型
├── config.yaml         # 数据集配置文件
├── main.py             # YoloV8训练脚本
├── requirements.txt    # 所需依赖包
├── shot_detector.py    # 投篮检测核心程序
├── utils.py            # 辅助工具文件
└── video_test_5.mp4    # 测试用视频
```

## 模型训练

训练过程使用了Ultralytics的YOLO实现，并使用了`config.yaml`文件中指定的自定义数据集。模型经过若干训练周期，最终保存下来的最佳模型权重将用于后续的投篮检测。虽然该模型适用于我的使用场景，但不同的数据集或训练方法可能会更适合你的特定项目需求。

## 算法原理

该项目的核心算法使用训练好的YOLOv8模型检测每一帧中的篮球和篮球架。算法分析篮球相对于篮球架的运动和位置，判断是否为成功投篮。

为了提高投篮检测的准确性，算法不仅追踪篮球的位置，还对篮球和篮球架的位置应用数据清洗技术。算法旨在过滤掉不准确的数据点，移除超过某一帧限制的数据，并防止在不同物体之间跳跃，从而保持检测的准确性。

通过线性回归算法预测篮球的轨迹，如果预测的轨迹与篮球架相交，算法就会注册为成功投篮。

## 安装与使用

### 快速开始（推荐）

1. 克隆本仓库：
```bash
git clone https://github.com/avishah3/AI-Basketball-Shot-Detector-Tracker.git
cd AI-Basketball-Shot-Detector-Tracker
```

2. 运行自动安装脚本：
```bash
chmod +x setup_basketball_ai.sh
./setup_basketball_ai.sh
```

3. 启动投篮检测器：
```bash
./run_detector.sh
```

### 手动安装

适用于高级用户或自定义设置：

1. 创建conda环境：
```bash
conda create -n basketball-ai python=3.8
conda activate basketball-ai
```

2. 安装Python依赖包：
```bash
pip install -r requirements.txt
pip install --upgrade ultralytics
```

3. 安装系统依赖：
```bash
sudo apt-get update
sudo apt-get install -y qt5-default libxcb-xinerama0 xvfb
```

4. 运行检测器：
```bash
xvfb-run --auto-servernum --server-args="-screen 0 1280x1024x24" python shot_detector.py
```

### 使用预训练模型

**如果不想自己训练模型**，可以直接使用提供的'best.pt'模型（跳过训练步骤）。

### 训练自定义模型（可选）

1. 下载config.yaml中指定的数据集
2. 调整配置文件中的路径
3. 按照main.py中的说明训练模型

### 输入选项
- 摄像头：默认输入（确保摄像头已连接）
- 视频文件：修改shot_detector.py指定视频路径

## 免责声明

模型的性能可能会根据视频源的质量、光照条件以及篮球和篮球架在视频中的清晰度而有所不同。此外，如果视频中有多个篮球和篮球架，程序将无法正常工作。在测试时，输入的视频是在户外拍摄的，使用的是手机摄像头拍摄的地面角度视频。