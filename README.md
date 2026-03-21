# 智能节能系统 (Smart Energy Saving System)

基于计算机视觉和深度学习的智能能源管理解决方案，包含**建筑智能节能**、**交通节能**和**计算机节能**三个主要方向。

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![YOLO12](https://img.shields.io/badge/YOLO-v12-green.svg)](https://github.com/ultralytics/ultralytics)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime-orange.svg)](https://onnxruntime.ai/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)
[![GraphRAG](https://img.shields.io/badge/RAG-GraphRAG-purple.svg)](https://github.com/microsoft/graphrag)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

---

## 🎯 项目简介

本项目旨在通过AI技术优化能源使用效率，涵盖建筑、交通、计算机三大领域，降低能耗成本，提升系统智能化水平。系统采用模块化设计，支持灵活部署和扩展。

### 核心功能
- **智能感知**：基于计算机视觉的人员/车辆检测和行为分析
- **预测控制**：深度学习驱动的能耗预测和优化控制
- **知识管理**：GraphRAG驱动的智能知识问答系统
- **异常监测**：基于数字孪生的设备故障检测和告警
- **强化学习**：RL驱动的智能决策优化（HVAC/信号灯/充电调度）

---

## 📋 功能模块

### 🏠 方向一：建筑智能节能系统 (`building_energy/`)

基于天气预报、数字孪生和强化学习的建筑能源管理系统。

#### 1. 天气集成与热负荷预测 (`data/weather_api.py`)
- **技术栈**: OpenWeatherMap API / WeatherAPI.com + PyTorch
- **功能**: 
  - 实时天气数据获取（温度、湿度、风速、太阳辐射）
  - 结合室内温度传感器数据
  - 基于建筑热力学模型预测热负荷
  - 动态调整空调运行策略

#### 2. 数字孪生温度调控 (`core/building_simulator.py`)
- **技术栈**: EnergyPlus + eppy + PINN (DeepXDE)
- **功能**:
  - 建筑能耗仿真（墙厚、面积、朝向、空调参数）
  - 物理信息神经网络温度场预测
  - 温度分布预测与最优控制策略
  - CFD数据训练代理模型

#### 3. 建筑储能智能调度 (`models/battery_scheduler.py`)
- **技术栈**: Stable-Baselines3 (SAC/PPO) + OR-Tools
- **功能**:
  - 基于峰谷电价的储能充放电优化
  - 电网电压/频率信号需求响应
  - 预测性储能调度（结合天气预报）
  - 预留电网API对接接口

#### 4. 智能照明改进 (`lighting/`)
- **技术栈**: Python + OpenCV
- **功能**:
  - 人因照明（Human-centric Lighting）：根据时间调节色温
  - 预测性开关：基于人员移动轨迹预测
  - 自然光自适应：结合光照传感器和窗帘控制

#### 5. 设备故障检测 (`models/fault_detector.py`)
- **技术栈**: PyOD + 自定义PID残差分析
- **功能**:
  - 基于数字孪生模型的正常运行基准
  - PID控制器残差分析
  - 多算法异常检测（Isolation Forest/Autoencoder）
  - 复用现有传感器，无需额外硬件

#### 6. 知识库模块 (`knowledge/graph_rag.py`)
- **技术栈**: Microsoft GraphRAG + Neo4j/YouTu Graph
- **功能**:
  - 文档自动解析（PDF/Markdown）
  - 语义向量检索
  - 知识图谱构建
  - 自然语言问答

---

### 🚗 方向二：交通节能系统 (`traffic_energy/`)

基于YOLO12和强化学习的智能交通能源管理系统。

#### 1. 车辆检测与跟踪 (`detection/`)
- **技术栈**: YOLO12 + BoT-SORT/ByteTrack
- **功能**: 
  - 车辆检测（轿车/SUV/卡车/公交车/电动车/燃油车分类）
  - 多目标跟踪（路径-时间图生成）
  - 车辆速度估计
  - Python自动化脚本处理数据

#### 2. 跨摄像头车辆匹配 (`reid/`)
- **技术栈**: FastReID + Milvus/PGVector
- **功能**:
  - 车辆外观特征提取（颜色、车型）
  - 跨摄像头轨迹关联
  - 余弦相似度匹配 + 时空约束
  - 基于车辆特征的匹配机制

#### 3. 交通信号优化 (`signal_opt/`)
- **技术栈**: SUMO + Stable-Baselines3 (PPO/SAC)
- **功能**:
  - 实时车流监测与预测
  - 动态红绿灯时长调节（RL控制）
  - 优先放行燃油车（减少怠速排放）
  - 路网拓扑建模（图神经网络）
  - 高德地图/交管部门数据对接（预留接口）

#### 4. 智能充电桩管理 (`charging/`)
- **技术栈**: OR-Tools + Prophet
- **功能**:
  - 基于车主日程规划的充电调度
  - 动态监测电网电压/频率
  - 错峰充电策略
  - 充电需求预测

---

### 💻 方向三：计算机节能系统 (`computer_energy/`)

基于LLM和进程监控的实验室计算机能源管理系统。

#### 1. 后台任务监控 (`monitor/`)
- **技术栈**: psutil + WMI (Windows) / procfs (Linux)
- **功能**: 
  - 定时获取进程列表（晚上9~11点每30分钟）
  - 白名单机制（系统常驻组件）
  - 资源占用分析（CPU/GPU/内存）
  - 支持10-20台电脑（主要为Win11）

#### 2. 智能关机决策 (`decision/`)
- **技术栈**: LLM API (OpenAI/本地Ollama)
- **功能**:
  - LLM辅助判断进程重要性
  - 判断是否有长时间负载任务
  - 通知用户（10分钟倒计时）
  - 支持远程登录取消关机

#### 3. CPU/GPU频率动态调节 (`power_manager/`)
- **技术栈**: powercfg + nvidia-ml-py
- **功能**:
  - 根据负载动态调节频率
  - 保持高boost响应能力
  - 长时间平均功耗降低
  - 待机/无重载任务设备降频

#### 4. 任务调度 (`scheduler/`)
- **技术栈**: APScheduler
- **功能**:
  - 定时执行节能策略
  - 定时扫描和决策
  - 分布式多机管理

---

## 🚀 快速开始

### 环境要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| Python | 3.10+ | 3.11+ |
| CUDA | 11.8+ (可选) | 12.1+ |
| RAM | 8GB | 16GB+ |
| GPU | GTX 1060 6GB | RTX 3060+ |

**各方向额外依赖：**
- **方向一**: EnergyPlus 9.6+, Neo4j 5.x (可选)
- **方向二**: SUMO 1.20+ (信号优化)
- **方向三**: Windows 11 / Linux (系统监控)

### 1. 克隆项目

```bash
git clone https://github.com/wdd9700/SmartEnergySavinginLightControlandACControl.git
cd SmartEnergySavinginLightControlandACControl
```

### 2. 安装依赖

**一键安装（推荐）：**
```bash
# 安装所有方向依赖
pip install -r requirements.txt
pip install -r building_energy/requirements.txt
pip install -r traffic_energy/requirements.txt
```

**分方向安装：**

```bash
# 方向一 - 建筑节能
pip install -r requirements.txt
pip install -r building_energy/requirements.txt

# 方向二 - 交通节能
pip install -r requirements.txt
pip install -r traffic_energy/requirements.txt

# 方向三 - 计算机节能
pip install -r requirements.txt
pip install -r computer_energy/requirements.txt
```

**边缘设备 (Jetson Nano)：**
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

#### 🏠 方向一：建筑智能节能

```bash
# 初始化配置
python -m building_energy.cli init

# 启动系统
python -m building_energy.cli start

# 查询知识库
python -m building_energy.cli query "如何优化空调能耗？"

# 训练强化学习模型
python building_energy/train_hvac_rl.py --algorithm SAC --timesteps 100000

# 查看系统状态
python -m building_energy.cli status
```

**功能验证：**
- ✅ 天气数据获取
- ✅ 数字孪生仿真
- ✅ 储能调度优化
- ✅ 设备故障检测
- ✅ GraphRAG知识问答

#### 🚗 方向二：交通节能

```bash
# 车辆检测与跟踪
python -m traffic_energy.cli detect --source traffic_video.mp4 --track

# 多摄像头处理
python -m traffic_energy.cli multi-camera --config config/cameras.yaml

# 交通信号优化训练
python -m traffic_energy.cli train-signal --algorithm PPO

# 充电调度优化
python -m traffic_energy.cli schedule-charging --vehicles 50

# 启动API服务
python -m traffic_energy.api.rest_api --port 8080
```

**功能验证：**
- ✅ YOLO12车辆检测 (≥30 FPS)
- ✅ BoT-SORT多目标跟踪 (MOTA ≥75%)
- ✅ 跨摄像头车辆匹配 (准确率 ≥85%)
- ✅ 流量统计 (误差 <5%)
- ✅ 信号优化 (等待时间减少 ≥15%)

#### 💻 方向三：计算机节能

```bash
# 启动监控守护进程
python -m computer_energy.cli monitor --daemon

# 手动执行节能检查
python -m computer_energy.cli check

# 配置节能策略
python -m computer_energy.cli config --save-config

# 查看监控报告
python -m computer_energy.cli report --last-24h
```

**功能验证：**
- ✅ 进程监控 (CPU/GPU/内存)
- ✅ LLM智能判断
- ✅ 自动关机通知
- ✅ 频率动态调节

### 5. 已有模块运行

**楼道灯控制系统：**
```bash
# Demo模式（仅显示，不控制硬件）
python -m corridor_light.main --source tests/test_corridor.mp4 --mode demo

# 使用摄像头
python -m corridor_light.main --source 0 --mode demo

# 区域模式（智能控制）
python -m corridor_light.main_unified --source 0 --mode zone_based --demo
```

**教室空调控制系统：**
```bash
python -m classroom_ac.main --source tests/test_classroom.mp4 --mode demo

# 热负荷计算模式
python -m classroom_ac.main_v3 --source 0 --outdoor-temp 32.0
```

**Web管理界面：**
```bash
# 启动Dashboard
python web/dashboard_http_server.py --port 8080

# 访问 http://localhost:8080
```

**数据记录与分析：**
```bash
# 查看数据文件
ls logs/
# detections_YYYYMMDD.csv - 检测记录
# events_YYYYMMDD.csv - 事件记录
# trajectories_YYYYMMDD.csv - 轨迹记录

# Web API查看统计
curl http://localhost:8080/status
curl http://localhost:8080/stats
curl http://localhost:8080/energy
```

### 5. 已有模块运行

**楼道灯控制系统:**
```bash
# Demo模式（仅显示，不控制硬件）
python -m corridor_light.main --source tests/test_corridor.mp4 --mode demo

# 使用摄像头
python -m corridor_light.main --source 0 --mode demo
```

**教室空调控制系统:**
```bash
python -m classroom_ac.main --source tests/test_classroom.mp4 --mode demo
```

**数据记录与分析:**
```bash
# 查看数据文件
ls logs/
# detections_YYYYMMDD.csv - 检测记录
# events_YYYYMMDD.csv - 事件记录

# Web API查看统计
curl http://localhost:8080/status
curl http://localhost:8080/stats
curl http://localhost:8080/energy
```

---

## 📁 项目结构

```
SmartEnergySavinginLightControlandACControl/
├── building_energy/          # 方向一：建筑智能节能系统
│   ├── main.py              # 主控制程序
│   ├── cli.py               # 命令行接口
│   ├── train_hvac_rl.py     # RL训练脚本
│   ├── config/              # 配置管理
│   ├── core/                # 核心模块
│   ├── data/                # 数据接口
│   ├── env/                 # RL环境
│   ├── knowledge/           # 知识库模块 (GraphRAG)
│   ├── models/              # 模型模块
│   ├── lighting/            # 智能照明改进
│   └── requirements.txt
│
├── traffic_energy/          # 方向二：交通节能系统
│   ├── detection/           # 车辆检测跟踪
│   ├── reid/                # 车辆重识别
│   ├── signal_opt/          # 信号优化
│   ├── charging/            # 充电桩管理
│   └── requirements.txt
│
├── computer_energy/         # 方向三：计算机节能系统
│   ├── monitor/             # 进程监控
│   ├── decision/            # 智能决策
│   ├── power_manager/       # 电源管理
│   ├── scheduler/           # 任务调度
│   └── requirements.txt
│
├── corridor_light/          # 楼道智能灯控 (已有模块)
├── classroom_ac/            # 教室空调控制 (已有模块)
├── shared/                  # 共享模块
├── web/                     # Web管理界面
├── tests/                   # 测试套件
├── docs/                    # 文档
├── models/                  # 预训练模型
└── innovations/             # 创新功能原型
```

---

## 🛠️ 技术栈

### 核心框架
| 类别 | 技术 | 说明 |
|------|------|------|
| 编程语言 | Python 3.10+ | 主开发语言 |
| 深度学习 | PyTorch 2.1+ | 神经网络训练 |
| 目标检测 | YOLO12 | 最新SOTA，注意力架构 |
| 多目标跟踪 | BoT-SORT / ByteTrack | 相机运动补偿 |
| 强化学习 | Stable-Baselines3 2.2+ | HVAC/信号灯/充电调度 |
| 建筑模拟 | EnergyPlus + eppy | 建筑能耗仿真 |
| 异常检测 | PyOD 1.1+ | 多算法异常检测 |
| RAG系统 | Microsoft GraphRAG | 知识图谱问答 |
| 图数据库 | Neo4j / YouTu Graph | 知识存储 |
| 向量检索 | FAISS / Milvus | 特征匹配 |
| 时序数据库 | TimescaleDB / InfluxDB | 传感器数据 |
| 优化求解 | OR-Tools / SciPy | 充电调度优化 |
| 文本嵌入 | Sentence-Transformers 2.2+ | 语义检索 |
| 计算机视觉 | OpenCV 4.8+ | 图像处理 |
| 系统监控 | psutil + WMI | 进程监控 |

### 推理优化
| 部署场景 | 推荐方案 | 延迟 |
|----------|----------|------|
| GPU生产环境 | TensorRT FP16 | ~1.5ms |
| CPU边缘设备 | OpenVINO INT8 | ~11ms |
| 跨平台部署 | ONNX Runtime | ~20ms |

### 主要依赖
```
# 基础依赖
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
pyyaml>=6.0
requests>=2.31.0

# 深度学习
torch>=2.1.0
torchvision>=0.16.0
ultralytics>=8.3.0  # YOLO12

# 强化学习
stable-baselines3>=2.2.0
gymnasium>=0.29.0

# 异常检测与机器学习
pyod>=1.1.0
scikit-learn>=1.3.0
prophet>=1.1.0

# RAG与知识图谱
graphrag>=0.3.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4
neo4j-python-driver>=5.14.0

# 建筑模拟
eppy>=0.5.63

# 优化求解
ortools>=9.8.0
scipy>=1.11.0

# 系统监控
psutil>=5.9.0
pywin32>=306; platform_system=="Windows"
nvidia-ml-py>=12.535.0

# 数据库
timescaledb>=1.0.0
sqlalchemy>=2.0.0

# 计算机视觉
opencv-python>=4.8.0
onnxruntime-gpu>=1.16.0; platform_system!="Darwin"
onnxruntime>=1.16.0; platform_system=="Darwin"

# 任务调度
apscheduler>=3.10.0
```

---

## 📊 系统架构

### 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                         应用层 (Application)                         │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │  建筑智能节能    │  │  交通节能        │  │  计算机节能      │     │
│  │  building_energy│  │  traffic_energy │  │ computer_energy │     │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘     │
├─────────────────────────────────────────────────────────────────────┤
│                         服务层 (Services)                            │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │   AI/ML服务   │  │   数据服务    │  │   知识服务    │              │
│  │  ├ YOLO检测   │  │  ├ 时序数据库  │  │  ├ GraphRAG  │              │
│  │  ├ 跟踪算法   │  │  ├ 向量数据库  │  │  ├ 向量检索   │              │
│  │  ├ 强化学习   │  │  ├ 关系数据库  │  │  ├ 知识图谱   │              │
│  │  └ 预测模型   │  │  └ 数据缓存   │  │  └ LLM接口   │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
├─────────────────────────────────────────────────────────────────────┤
│                      基础设施层 (Infrastructure)                      │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │   硬件接口    │  │   网络通信    │  │   系统监控    │              │
│  │  ├ 传感器    │  │  ├ REST API  │  │  ├ 日志系统   │              │
│  │  ├ 执行器    │  │  ├ WebSocket │  │  ├ 性能监控   │              │
│  │  └ 摄像头    │  │  └ MQTT      │  │  └ 告警系统   │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
```

### 数据流

```
传感器/摄像头 → 检测/跟踪 → 数据分析 → 决策优化 → 执行控制
     ↓              ↓            ↓            ↓            ↓
   原始数据      结构化数据    统计信息     控制指令     节能效果
   (视频/温度)   (位置/轨迹)   (流量/能耗)  (开关/调节)  (节省kWh)
```

详见 [docs/architecture/diagram.md](docs/architecture/diagram.md)

## 📚 开发指导文档

### 需求与设计文档

| 文档 | 内容 | 状态 |
|------|------|------|
| [项目需求文档](docs/project_requirements.md) | 三个方向的详细功能需求 | ✅ 已完成 |
| [基础设施复用分析](docs/infrastructure_reuse_analysis.md) | 三个方向可复用组件分析 | ✅ 已完成 |
| [亮度校准架构](docs/brightness_calibration_architecture.md) | 楼道灯控亮度校准设计 | ✅ 已完成 |
| [热控制架构](docs/thermal_control_v5.md) | 空调热控制架构设计 | ✅ 已完成 |

### 技术栈调研

| 文档 | 内容 | 状态 |
|------|------|------|
| [方向一技术栈调研](docs/direction1_tech_stack_research.md) | 建筑节能技术选型与实现方案 | ✅ 已完成 |
| [方向二技术栈调研](docs/direction2_tech_stack_research.md) | 交通节能技术选型与实现方案 | ✅ 已完成 |
| [方向二开发指导](docs/direction2_development_guide.md) | 交通节能开发详细指导 | ✅ 已完成 |
| [方向三技术栈调研](docs/direction3_tech_stack_research_computer.md) | 计算机节能技术选型与实现方案 | 🔄 待创建 |

### Subagent 任务文档

| 文档 | 内容 | 用途 |
|------|------|------|
| [.agents/TASKS_DIRECTION2.md](.agents/TASKS_DIRECTION2.md) | 方向二详细任务分解 | Subagent开发指导 |
| [.agents/PROMPT_DIRECTION2_QUICK.md](.agents/PROMPT_DIRECTION2_QUICK.md) | 快速启动Prompt模板 | 快速启动Subagent |

---

## 📈 性能基准

### 方向一：建筑智能节能

| 指标 | 目标值 | 测试条件 |
|------|--------|----------|
| 热负荷预测误差 | < 5% | 24小时预测 |
| RL训练收敛 | < 100k steps | SAC算法 |
| 故障检测准确率 | > 90% | 异常场景 |
| 知识库查询延迟 | < 500ms | 典型查询 |

### 方向二：交通节能

| 指标 | 目标值 | 测试条件 |
|------|--------|----------|
| 检测帧率 | ≥ 30 FPS | YOLO12n @ 640x480 |
| 跟踪MOTA | ≥ 75% | MOT17数据集 |
| 跨摄像头匹配 | ≥ 85% | 同车型匹配 |
| 流量统计误差 | < 5% | 虚拟线圈计数 |
| 信号优化效果 | ≥ 15% | 平均等待时间减少 |
| 充电成本降低 | ≥ 20% | 峰谷电价优化 |

### 方向三：计算机节能

| 指标 | 目标值 | 测试条件 |
|------|--------|----------|
| 进程识别准确率 | > 95% | LLM判断 |
| 误关机率 | < 1% | 重要任务保护 |
| 功耗降低 | ≥ 15% | 频率调节 |

---

## 🤝 贡献指南

### 开发流程

1. **Fork 本仓库**
2. **创建特性分支** (`git checkout -b feature/AmazingFeature`)
3. **开发并测试** - 确保通过所有测试
4. **提交更改** (`git commit -m 'Add some AmazingFeature'`)
5. **推送到分支** (`git push origin feature/AmazingFeature`)
6. **创建 Pull Request** - 描述改动和测试情况

### 代码规范

- **风格**: 遵循 PEP 8，使用 Black 格式化
- **类型**: 所有函数添加类型注解
- **文档**: Google风格docstring
- **测试**: 单元测试覆盖率 ≥ 80%
- **提交**: 遵循 Conventional Commits 规范

### 提交信息格式

```
<type>(<scope>): <subject>

<body>

<footer>
```

**类型说明：**
- `feat`: 新功能
- `fix`: 修复bug
- `docs`: 文档更新
- `style`: 代码格式（不影响功能）
- `refactor`: 重构
- `test`: 测试相关
- `chore`: 构建/工具

**示例：**
```
feat(detection): 添加YOLO12车辆检测器

- 支持car/truck/bus/motorcycle分类
- 集成TensorRT加速
- 添加批量推理接口

Closes #123
```

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 📞 联系方式

如有问题或建议，欢迎提交 Issue 或 Pull Request。

---

## ❓ 常见问题

### 安装问题

**Q: 安装ultralytics时出现依赖冲突？**
```bash
# 解决方案：使用conda环境
conda create -n smart_energy python=3.11
conda activate smart_energy
pip install -r requirements.txt
```

**Q: ONNX Runtime GPU版本安装失败？**
```bash
# 解决方案：根据CUDA版本选择
# CUDA 11.8
pip install onnxruntime-gpu==1.16.3 --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/

# CUDA 12.1
pip install onnxruntime-gpu==1.16.3
```

### 运行问题

**Q: YOLO12检测速度慢？**
```bash
# 优化方案1：使用TensorRT
python -c "from ultralytics import YOLO; YOLO('yolo12n.pt').export(format='engine')"

# 优化方案2：降低分辨率
python -m traffic_energy.cli detect --source video.mp4 --imgsz 480

# 优化方案3：使用OpenVINO (CPU)
python -c "from ultralytics import YOLO; YOLO('yolo12n.pt').export(format='openvino', int8=True)"
```

**Q: SUMO仿真环境启动失败？**
```bash
# 检查SUMO安装
sumo --version

# 设置环境变量
export SUMO_HOME=/usr/share/sumo
export PATH=$PATH:$SUMO_HOME/bin

# Windows
set SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo
```

**Q: GPU内存不足？**
```bash
# 解决方案1：使用更小的模型
# yolo12n.pt (nano) < yolo12s.pt (small) < yolo12m.pt (medium)

# 解决方案2：降低batch size
python -m traffic_energy.cli detect --batch-size 1

# 解决方案3：使用CPU推理
python -m traffic_energy.cli detect --device cpu
```

### 开发问题

**Q: 如何添加新的车辆类型？**
```python
# 修改 traffic_energy/detection/vehicle_detector.py
VEHICLE_CLASSES = {
    2: {"name": "car", "fuel_type": "unknown"},
    3: {"name": "motorcycle", "fuel_type": "gasoline"},
    5: {"name": "bus", "fuel_type": "diesel"},
    7: {"name": "truck", "fuel_type": "diesel"},
    # 添加新类型
    8: {"name": "van", "fuel_type": "gasoline"},
}
```

**Q: 如何调试跨摄像头匹配？**
```bash
# 启用调试模式
python -m traffic_energy.cli match --debug --save-visualization

# 查看匹配日志
tail -f logs/matcher.log
```

### 部署问题

**Q: 如何部署到边缘设备？**
```bash
# Jetson Nano
pip install -r requirements-jetson.txt
python -c "from ultralytics import YOLO; YOLO('yolo12n.pt').export(format='engine', half=True)"

# Raspberry Pi
pip install opencv-python-headless
pip install -r requirements.txt --no-deps
```

**Q: 如何配置多摄像头？**
```yaml
# 编辑 config/camera_topology.yaml
cameras:
  - id: "cam_001"
    url: "rtsp://192.168.1.101:554/stream"
    location: [116.3974, 39.9042]  # 经纬度
    direction: "north"
  - id: "cam_002"
    url: "rtsp://192.168.1.102:554/stream"
    location: [116.3980, 39.9050]
    direction: "south"
```

---

## 🙏 致谢

### 核心框架
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - 强化学习框架
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - 目标检测 (YOLO12)
- [Microsoft GraphRAG](https://github.com/microsoft/graphrag) - 知识图谱RAG
- [EnergyPlus](https://energyplus.net/) - 建筑能耗模拟

### 交通节能
- [BoT-SORT](https://github.com/NirAharon/BoT-SORT) - 多目标跟踪
- [FastReID](https://github.com/JDAI-CV/fast-reid) - 车辆重识别
- [SUMO](https://www.eclipse.org/sumo/) - 交通仿真
- [ByteTrack](https://github.com/ifzhang/ByteTrack) - 多目标跟踪

### 异常检测与优化
- [PyOD](https://pyod.readthedocs.io/) - 异常检测库
- [Google OR-Tools](https://developers.google.com/optimization) - 优化求解器
- [Prophet](https://facebook.github.io/prophet/) - 时间序列预测

### 系统监控
- [psutil](https://github.com/giampaolo/psutil) - 系统监控
- [APScheduler](https://apscheduler.readthedocs.io/) - 任务调度
