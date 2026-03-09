# 智能节能控制系统
# Intelligent Energy-Saving Control System

## 项目结构

```
smart-energy/
├── corridor_light/          # 项目一：楼道灯智能控制
│   ├── config.yaml         # 配置文件
│   ├── main.py             # 主程序入口
│   ├── detector.py         # 人形检测模块
│   ├── enhancer.py         # 视频增强模块
│   ├── controller.py       # 灯光控制模块
│   └── utils.py            # 工具函数
├── classroom_ac/           # 项目二：教室空调智能控制
│   ├── config.yaml
│   ├── main.py
│   ├── people_counter.py   # 人流统计模块
│   ├── zone_manager.py     # 区域管理模块
│   ├── ac_controller.py    # 空调控制模块
│   └── utils.py
├── models/                 # 预训练模型存放
│   ├── yolov8n.onnx       # YOLOv8 nano ONNX模型
│   └── download_models.py  # 模型下载脚本
├── shared/                 # 共享模块
│   ├── video_capture.py    # 视频捕获封装
│   ├── mqtt_client.py      # MQTT通信（可选）
│   └── logger.py           # 日志模块
├── tests/                  # 测试视频/脚本
│   └── test_video.mp4     # 测试用视频
└── requirements.txt        # 依赖清单
```

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 下载模型
```bash
cd models
python download_models.py
```

### 3. 运行测试
```bash
# 项目一：楼道灯
python -m corridor_light.main --source tests/test_video.mp4 --mode demo

# 项目二：空调控制
python -m classroom_ac.main --source tests/test_video.mp4 --mode demo
```

## 硬件连接（实际部署）

### 楼道灯控制
- GPIO 17 → 继电器模块 → 灯光电路
- USB摄像头 → 树莓派/Jetson

### 空调控制
- GPIO 18 → 红外发射模块 或 继电器 → 空调
- USB摄像头覆盖教室入口

## 配置说明

见各项目的 `config.yaml` 文件，可根据实际场景调整：
- 检测灵敏度
- 延迟时间
- 控制逻辑阈值
