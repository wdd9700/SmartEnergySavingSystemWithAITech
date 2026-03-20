# 建筑智能节能系统 (Building Energy Management System)

基于计算机视觉和深度学习的建筑能源管理解决方案，包含**楼道智能灯控**、**教室空调智能调节**和**建筑能耗优化**三个子系统。

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-green.svg)](https://github.com/ultralytics/ultralytics)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime-orange.svg)](https://onnxruntime.ai/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

---

## 🎯 项目简介

本项目旨在通过智能化手段优化建筑能源使用效率，降低能耗成本，提升用户舒适度。系统采用模块化设计，支持灵活部署和扩展。

### 核心功能
- **智能感知**：基于计算机视觉的人员检测和行为分析
- **预测控制**：深度学习驱动的能耗预测和优化控制
- **知识管理**：GraphRAG驱动的能耗知识问答系统
- **异常监测**：基于PyOD的设备异常检测和告警

---

## 📋 功能模块

### 🏠 方向一：建筑能耗优化系统 (`building_energy/`)

#### 1. 异常检测模块 (`models/anomaly_detector.py`)
- **技术栈**: PyOD + scikit-learn
- **功能**: 
  - 多算法异常检测（Isolation Forest, LOF, ABOD等）
  - 实时数据流异常监测
  - 自适应阈值调整
  - 告警分级管理

#### 2. 知识库模块 (`knowledge/graph_rag.py`)
- **技术栈**: Sentence-Transformers + FAISS + GraphRAG
- **功能**:
  - 文档自动解析（PDF/Markdown）
  - 语义向量检索
  - 知识图谱构建
  - 自然语言问答

#### 3. 预测模型模块 (`models/predictor.py`)
- **技术栈**: PyTorch + LSTM/Transformer
- **功能**:
  - 多变量时间序列预测
  - 能耗趋势分析
  - 不确定性量化
  - 模型自动重训练

#### 4. 强化学习优化 (`env/hvac_env.py`, `train_hvac_rl.py`)
- **技术栈**: Stable-Baselines3 + Gymnasium
- **功能**:
  - HVAC控制策略优化
  - SAC/PPO/TD3算法支持
  - EnergyPlus建筑模拟集成
  - 能耗-舒适度平衡优化

#### 5. 主控制程序 (`main.py`, `cli.py`)
- **功能**:
  - 系统生命周期管理
  - 模块协调调度
  - CLI交互界面
  - 实时监控和告警

---

### 🚗 方向二：交通节能系统 (`traffic_energy/` 待开发)
- **车辆检测与跟踪**：YOLO12 + BoT-SORT
- **跨摄像头车辆匹配**：FastReID 车辆重识别
- **交通流量分析**：虚拟线圈检测、速度估计
- **信号灯配时优化**：强化学习 (Stable-Baselines3) 动态调节
- **智能充电桩管理**：充电需求预测、错峰调度

> **注意**：方向二（原楼道灯控 `corridor_light/`）已整合到方向一作为子模块

---

### 💻 方向三：计算机节能系统 (`computer_energy/` 待开发)
- **进程监控**：psutil 定时扫描进程列表
- **智能关机决策**：LLM 辅助判断进程重要性，自动关机
- **CPU/GPU 频率调节**：动态调频降低功耗
- **通知系统**：关机前通知用户，支持取消
- **任务调度**：APScheduler 定时执行节能策略

> **注意**：方向三（原教室空调 `classroom_ac/`）已整合到方向一作为子模块

---

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone <repository-url>
cd SmartEnergySavinginLightControlandACControl

# 安装方向一依赖
pip install -r building_energy/requirements.txt

# 安装方向二/三依赖（可选）
pip install -r requirements.txt
```

### 使用示例

```bash
# 初始化配置
python -m building_energy.cli init

# 启动系统
python -m building_energy.cli start

# 使用指定配置启动
python -m building_energy.cli start -c config.yaml

# 查询知识库
python -m building_energy.cli query "如何优化空调能耗？"

# 查看系统状态
python -m building_energy.cli status

# 查看告警
python -m building_energy.cli alerts

# 停止系统
python -m building_energy.cli stop
```

### 训练强化学习模型

```bash
# 快速测试训练
python building_energy/test_train_quick.py

# 完整训练流程
python building_energy/train_hvac_rl.py \
    --algorithm SAC \
    --timesteps 100000 \
    --output-dir ./models/rl
```

---

## 📁 项目结构

```
SmartEnergySavinginLightControlandACControl/
├── building_energy/          # 方向一：建筑能耗优化系统
│   ├── main.py              # 主控制程序
│   ├── cli.py               # 命令行接口
│   ├── train_hvac_rl.py     # RL训练脚本
│   ├── config/              # 配置管理
│   │   ├── default_config.yaml
│   │   └── manager.py
│   ├── core/                # 核心模块
│   │   └── building_simulator.py
│   ├── data/                # 数据接口
│   │   └── weather_api.py
│   ├── env/                 # RL环境
│   │   └── hvac_env.py
│   ├── knowledge/           # 知识库模块
│   │   ├── graph_rag.py
│   │   ├── document_loader.py
│   │   └── example_usage.py
│   ├── models/              # 模型模块
│   │   ├── anomaly_detector.py
│   │   ├── predictor.py
│   │   └── baseline.py
│   ├── visualization/       # 可视化
│   │   └── plots.py
│   └── requirements.txt     # 依赖列表
│
├── traffic_energy/          # 方向二：交通节能系统 (待开发)
│   ├── detection/           # 车辆检测跟踪
│   ├── reid/                # 车辆重识别
│   ├── signal_opt/          # 信号优化
│   └── charging/            # 充电桩管理
│
├── computer_energy/         # 方向三：计算机节能系统 (待开发)
│   ├── monitor/             # 进程监控
│   ├── power_manager/       # 电源管理
│   └── scheduler/           # 任务调度
│
├── corridor_light/          # [整合到方向一] 楼道智能灯控
│   ├── main.py, main_v3.py, main_unified.py
│   ├── detector.py          # 人形检测
│   ├── zone_controller.py   # 区域控制
│   ├── brightness_analyzer.py
│   ├── auto_calibrator.py
│   └── config.yaml
│
├── classroom_ac/            # [整合到方向一] 教室空调控制
│   ├── main.py, main_v3.py
│   ├── thermal_controller.py
│   ├── people_counter.py
│   ├── ac_controller.py
│   └── config.yaml
│
├── shared/                  # 共享模块
│   ├── video_capture.py
│   ├── data_recorder.py
│   ├── jetson_optimizer.py
│   └── logger.py
│
├── web/                     # Web管理界面
│   ├── dashboard.html
│   ├── dashboard_server.py
│   └── templates/
│
├── tests/                   # 测试套件
│   ├── test_suite.py
│   ├── test_anomaly_detector.py
│   ├── test_predictor.py
│   └── test_knowledge_base.py
│
├── docs/                    # 文档
│   ├── architecture/        # 架构文档
│   ├── direction1_tech_stack_research.md
│   ├── direction2_tech_stack_research.md
│   └── direction3_tech_stack_research.md
│
├── models/                  # 预训练模型
├── innovations/             # 创新功能原型
└── README.md               # 本文件
```

---

## 🛠️ 技术栈

### 核心框架
| 类别 | 技术 |
|------|------|
| 编程语言 | Python 3.10+ |
| 深度学习 | PyTorch 2.1+ |
| 强化学习 | Stable-Baselines3 2.2+ |
| 异常检测 | PyOD 1.1+ |
| 向量检索 | FAISS 1.7+ |
| 文本嵌入 | Sentence-Transformers 2.2+ |
| 计算机视觉 | YOLOv8, OpenCV |
| 建筑模拟 | EnergyPlus (可选) |

### 主要依赖
```
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
pyyaml>=6.0
stable-baselines3>=2.2.0
gymnasium>=0.29.0
pyod>=1.1.0
scikit-learn>=1.3.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4
```

---

## 📊 系统架构

详见 [docs/architecture/diagram.md](docs/architecture/diagram.md)

---

## 🤝 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

### 代码规范
- 遵循 PEP 8 代码风格
- 使用 Black 进行代码格式化
- 添加类型注解
- 编写单元测试

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 📞 联系方式

如有问题或建议，欢迎提交 Issue 或 Pull Request。

---

## 🙏 致谢

- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [PyOD](https://pyod.readthedocs.io/)
- [YOLOv8](https://github.com/ultralytics/ultralytics)
- [EnergyPlus](https://energyplus.net/)
│   ├── DEPLOYMENT.md
│   ├── HARDWARE.md
│   ├── brightness_calibration_architecture.md
│   └── thermal_control_v5.md
├── requirements.txt
├── requirements-jetson.txt
└── README.md
```

---

## 🚀 快速开始

### 环境要求
- Python 3.10+
- OpenCV 4.8+
- ONNX Runtime 1.15+
- (可选) CUDA 11.x 用于GPU加速

### 1. 克隆项目
```bash
git clone https://github.com/wdd9700/SmartEnergySavinginLightControlandACControl.git
cd SmartEnergySavinginLightControlandACControl
```

### 2. 安装依赖

**通用 Linux/Windows:**
```bash
pip install -r requirements.txt
```

**Jetson Nano (ARM64):**
```bash
pip install -r requirements-jetson.txt
```

### 3. 下载模型
```bash
cd models
python download_models.py
cd ..
```

### 4. 运行演示

**生成测试视频:**
```bash
python tests/create_test_videos.py
```

**运行楼道灯控制系统:**
```bash
# Demo模式（仅显示，不控制硬件）
python -m corridor_light.main --source tests/test_corridor.mp4 --mode demo

# 使用摄像头
python -m corridor_light.main --source 0 --mode demo
```

**运行教室空调控制系统:**
```bash
python -m classroom_ac.main --source tests/test_classroom.mp4 --mode demo
```

### 5. 高级功能

**统一主程序 (支持模式切换):**
```bash
# 传统模式
python -m corridor_light.main_unified --source 0 --mode traditional --demo

# 区域模式 (基于位置的智能控制)
python -m corridor_light.main_unified --source 0 --mode zone_based --demo

# 运行时按 'm' 键切换模式
```

**数据记录与分析:**
```bash
# 启用数据记录 (默认启用)
python -m corridor_light.main_unified --source 0 --mode zone_based

# 查看数据文件
ls logs/
# detections_YYYYMMDD.csv - 检测记录
# events_YYYYMMDD.csv - 事件记录
# heatmap_*.jpg - 人员热力图

# Web API查看统计
curl http://localhost:8080/status
curl http://localhost:8080/stats
curl http://localhost:8080/energy
```

**灯光自动校准 (v4.0):**
```bash
# 使用亮度分析自动确定灯光位置
python -m corridor_light.auto_calibrator --demo

# 查看架构文档
cat docs/brightness_calibration_architecture.md
```

**热负荷计算与预测性控制 (v5.0):**
```bash
# 基于热负荷的空调控制
python -m classroom_ac.main_v3 --source 0 --outdoor-temp 32.0

# 查看文档
cat docs/thermal_control_v5.md
```

**Web管理界面 (v6.0):**
```bash
# 启动管理界面服务器
python web/dashboard_http_server.py --port 8080

# 远程访问 (替换为服务器实际IP)
http://服务器IP:8080
```

---

## 🔧 配置说明

### 楼道灯控制 (`corridor_light/config.yaml`)
```yaml
demo_mode: true              # true=仅显示，false=控制硬件
conf_threshold: 0.5          # 检测置信度 (0-1)
brightness_threshold: 50     # 低光照阈值 (0-255)
enhance_method: "clahe"      # 增强算法: clahe|gamma|msrcr
light_off_delay: 5.0         # 关灯延迟 (秒)
gpio_pin: 17                 # GPIO引脚
```

### 教室空调控制 (`classroom_ac/config.yaml`)
```yaml
demo_mode: true
min_people: 3                # 开空调最少人数
cooldown_minutes: 5          # 冷却时间 (分钟)
zones:                       # 检测区域定义
  - name: "Front"
    coords: [[0,0], [0.5,0], [0.5,1], [0,1]]
  - name: "Back"
    coords: [[0.5,0], [1,0], [1,1], [0.5,1]]
```

---

## 🖥️ 部署方案

### 方案一：通用 Linux/Windows 部署

#### 系统要求
- CPU: x86_64, 2核+
- 内存: 4GB+
- 摄像头: USB摄像头或IP摄像头

#### 安装步骤
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3-pip python3-opencv libopencv-dev

# 安装Python依赖
pip3 install --user -r requirements.txt

# 验证安装
python3 -c "import cv2; import onnxruntime; print('OK')"
```

#### 开机自启动 (Systemd)
```bash
# 创建服务文件
sudo tee /etc/systemd/system/corridor-light.service << 'EOF'
[Unit]
Description=Corridor Light Control
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/smart-energy
ExecStart=/usr/bin/python3 -m corridor_light.main --mode deploy
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable corridor-light
sudo systemctl start corridor-light
```

### 方案二：Jetson Nano 部署

#### 硬件准备
- Jetson Nano 4GB/2GB
- 摄像头: Raspberry Pi Camera v2 或 USB摄像头
- 继电器模块 x2
- 杜邦线若干

#### 系统准备
```bash
# 1. 安装 JetPack (已包含CUDA/cuDNN/TensorRT)
# 2. 设置 swap (防止OOM)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 安装依赖
```bash
# Jetson 专用 ONNX Runtime
wget https://nvidia.box.com/shared/static/jy7nqxa8z1mlqnvkhhepqp6y7t8d22vg.whl -O onnxruntime_gpu.whl
pip3 install onnxruntime_gpu.whl

# 安装其他依赖
pip3 install -r requirements-jetson.txt
```

#### 性能优化
```bash
# 启用 MAXN 模式 (最高性能)
sudo nvpmodel -m 0
sudo jetson_clocks

# 验证GPU可用
python3 -c "import onnxruntime as ort; print(ort.get_device())"
```

#### GPIO 接线
```
Jetson Nano GPIO:
  Pin 17 (GPIO_17) → 继电器1 (楼道灯)
  Pin 18 (GPIO_18) → 继电器2 (空调)
  GND → 继电器GND
  5V → 继电器VCC (如需要)
```

### 方案三：树莓派部署

#### 系统要求
- Raspberry Pi 4B (2GB+)
- Raspberry Pi OS (64-bit)
- USB摄像头

#### 安装步骤
```bash
# 启用摄像头
sudo raspi-config  # Interface Options → Camera → Enable

# 安装依赖
sudo apt install -y python3-opencv python3-pip libcamera-dev
pip3 install --user -r requirements.txt

# 安装GPIO库
pip3 install RPi.GPIO
```

---

## 📊 性能指标

| 平台 | 模型 | 分辨率 | FPS | 延迟 |
|------|------|--------|-----|------|
| x86_64 CPU | YOLOv8n | 640x480 | 15-20 | 50-70ms |
| Jetson Nano GPU | YOLOv8n | 640x480 | 20-25 | 40-50ms |
| 树莓派4 | YOLOv8n | 320x240 | 8-10 | 100ms |

---

## 🔌 硬件连接图

### 楼道灯控制接线
```
┌─────────────────┐
│   USB Camera    │
└────────┬────────┘
         │ USB
┌────────▼────────┐      ┌──────────┐      ┌─────────┐
│  Jetson/RPi    │───────│ 继电器模块 │──────│ 楼道灯  │
│  (GPIO 17)     │      └──────────┘      └─────────┘
└─────────────────┘
```

### 空调控制接线 (红外方案)
```
┌─────────────────┐
│   USB Camera    │
└────────┬────────┘
         │ USB
┌────────▼────────┐      ┌──────────┐      ┌─────────┐
│  Jetson/RPi    │───────│ 红外发射管│──────│  空调   │
│  (GPIO 23)     │      └──────────┘      └─────────┘
└─────────────────┘
```

---

## 🛠️ 故障排查

### 常见问题

**Q: 摄像头无法打开**
```bash
# 检查设备
ls /dev/video*
v4l2-ctl --list-devices

# 检查权限
sudo usermod -a -G video $USER
```

**Q: 模型加载失败**
```bash
# 手动下载模型
cd models
wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.onnx
```

**Q: GPIO权限错误**
```bash
# 添加到gpio组
sudo usermod -a -G gpio $USER
```

**Q: Jetson上CUDA不可用**
```bash
# 检查CUDA
/usr/local/cuda/bin/nvcc --version

# 设置环境变量
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
```

---

## 📝 API 文档

### 程序参数

**corridor_light.main:**
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--source` | str | '0' | 视频源: 0=摄像头, 路径=文件 |
| `--mode` | str | 'demo' | demo/deploy |
| `--conf` | float | 0.5 | 检测置信度 |
| `--off-delay` | float | 5.0 | 关灯延迟(秒) |

**classroom_ac.main:**
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--source` | str | '0' | 视频源 |
| `--mode` | str | 'demo' | demo/deploy |
| `--min-people` | int | 3 | 开空调最少人数 |
| `--cooldown` | int | 5 | 冷却时间(分钟) |

---

## 📈 版本历史

### v6.0 - Web管理界面
- 纯HTML管理界面，支持远程访问
- 灯光位置、摄像头位置可视化
- 人员热力图、热负荷分析、功耗统计
- 实时控制决策日志

### v5.0 - 热负荷计算与预测性控制
- 人体活动状态产热计算 (休息/思考/运动)
- 电子设备产热检测 (笔记本20W)
- 围护结构传热、太阳辐射计算
- 课表驱动的提前预冷/预热
- 历史趋势分析与负荷预测

### v4.0 - 亮度分析自动校准
- 基于亮度分析的灯光位置自动校准
- 多摄像头相对位置校准
- 等亮度线分析、照明半径估算
- 无需人工测量的自动定位

### v3.0 - 基于位置的灯光控制
- 人形脚底位置提取
- 灯光区域配置管理
- 只开启人所在和人前方的灯
- 人离开区域后立即关灯

### v2.0 - 系统增强
- 多节点协调
- Jetson优化
- Web Dashboard基础
- 能耗分析

### v1.0 - 基础功能
- 楼道灯智能控制
- 教室空调控制
- YOLOv8人形检测
- 视频增强

---

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE)

---

## 🤝 贡献

欢迎提交 Issue 和 PR！

---

## 📧 联系

如有问题，请通过 GitHub Issues 联系。
