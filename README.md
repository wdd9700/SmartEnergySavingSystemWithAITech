# 智能节能控制系统
# Intelligent Energy-Saving Control System

基于计算机视觉的校园节能解决方案，包含**楼道智能灯控**和**教室空调智能调节**两个子系统。

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-green.svg)](https://github.com/ultralytics/ultralytics)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime-orange.svg)](https://onnxruntime.ai/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

---

## 📋 功能特性

### 🏠 楼道智能灯控系统 (Corridor Light Control)
- **低光照视频增强**：CLAHE/Gamma/MSRCR 三种算法自动切换
- **人形检测**：YOLOv8-nano，6MB模型，CPU实时推理 (9.5 FPS)
- **多种控制模式**：
  - 传统模式：单灯整体控制
  - **区域模式**：基于人形位置的区间控制，只开人所在和人前方的灯
  - **自动校准**：基于亮度分析自动确定灯位置 (v4.0)
- **智能控制**：人来灯亮、人走灯灭，支持延迟配置
- **数据记录**：检测日志、人员热力图、能耗统计
- **隐私保护**：本地处理，不上传视频

### 🏫 教室空调智能调节系统 (Classroom AC Control)
- **人流密度分析**：多区域热力图可视化
- **热负荷计算**：
  - 人体产热: 休息80W/人, 思考100W/人, 轻度运动150W/人
  - 设备产热: 笔记本20W, 台式机80W (基于实际功率)
  - 围护结构传热 + 太阳辐射
- **智能启停**：根据人数自动开关空调
- **功率调节**：人多降温、人少节能
- **预测性控制**：
  - 基于课表提前10分钟预冷
  - 历史人数趋势分析
  - 10分钟热负荷预测
- **防频繁切换**：冷却时间保护机制

### 🖥️ Web管理界面 (v6.0)
- **远程访问**：纯HTML界面，无需Flask依赖
- **实时监控**：
  - 校准后灯光位置可视化
  - 多摄像头相对位置显示
  - 人员位置热力图
- **热负荷分析**：制热/制冷量实时计算与分解
- **功耗统计**：空调/风扇功耗、累计能耗、成本估算
- **控制决策日志**：实时显示控制策略

---

## 📁 项目结构

```
smart-energy/
├── corridor_light/          # 楼道灯智能控制
│   ├── config.yaml         # 配置文件
│   ├── main.py             # 主程序入口 (传统模式)
│   ├── main_v3.py          # 主程序v3 (区域模式)
│   ├── main_unified.py     # 统一主程序 (支持模式切换)
│   ├── detector.py         # 人形检测模块 (含脚底位置提取)
│   ├── controller.py       # 传统灯光控制模块
│   ├── zone_controller.py  # 区域灯光控制模块 (v3)
│   ├── light_zones.py      # 灯光区域配置管理
│   ├── brightness_analyzer.py  # 亮度分析模块 (v4)
│   ├── auto_calibrator.py  # 自动校准工具 (v4)
│   ├── multi_camera_calibrator.py  # 多摄像头校准 (v4)
│   ├── enhancer.py         # 视频增强模块
│   └── __init__.py
├── classroom_ac/           # 教室空调智能控制
│   ├── config.yaml
│   ├── main.py             # 主程序v1/v2
│   ├── main_v3.py          # 主程序v3 (热负荷计算版)
│   ├── thermal_controller.py  # 热负荷计算与预测控制 (v5)
│   ├── schedule.json       # 课表配置
│   ├── people_counter.py   # 人流统计模块
│   ├── zone_manager.py     # 区域管理模块
│   ├── ac_controller.py    # 空调控制模块
│   └── __init__.py
├── shared/                 # 共享模块
│   ├── video_capture.py    # 视频捕获封装
│   ├── data_recorder.py    # 数据记录与分析
│   ├── coordination.py     # 多节点协调
│   ├── jetson_optimizer.py # Jetson优化
│   ├── environment.py      # 环境感知
│   └── logger.py           # 日志模块
├── web/                    # Web管理界面 (v6.0)
│   ├── dashboard.html      # 管理界面主页面
│   ├── dashboard_http_server.py  # HTTP服务器
│   └── templates/          # Flask模板 (可选)
├── models/                 # 预训练模型
│   └── download_models.py  # 模型下载脚本
├── tests/                  # 测试套件
│   ├── test_suite.py
│   ├── test_zone_light.py
│   ├── test_brightness_calibration.py
│   ├── test_data_recorder.py
│   ├── test_thermal_control.py
│   └── test_dashboard.py
├── docs/                   # 文档
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
