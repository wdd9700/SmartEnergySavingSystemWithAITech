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
- Python 3.10+
- OpenCV 4.8+
- ONNX Runtime 1.15+
- (可选) CUDA 11.x 用于GPU加速
- (可选) EnergyPlus 9.6+ 用于建筑模拟
- (可选) Neo4j 5.x 用于知识图谱

### 1. 克隆项目
```bash
git clone https://github.com/wdd9700/SmartEnergySavinginLightControlandACControl.git
cd SmartEnergySavinginLightControlandACControl
```

### 2. 安装依赖

**方向一 - 建筑节能（完整安装）:**
```bash
pip install -r requirements.txt
pip install -r building_energy/requirements.txt
```

**方向二 - 交通节能:**
```bash
pip install -r requirements.txt
pip install -r traffic_energy/requirements.txt
```

**方向三 - 计算机节能:**
```bash
pip install -r requirements.txt
pip install -r computer_energy/requirements.txt
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

#### 方向一：建筑智能节能

```bash
# 初始化配置
python -m building_energy.cli init

# 启动系统
python -m building_energy.cli start

# 查询知识库
python -m building_energy.cli query "如何优化空调能耗？"

# 训练强化学习模型
python building_energy/train_hvac_rl.py --algorithm SAC --timesteps 100000
```

#### 方向二：交通节能（开发中）

```bash
# 车辆检测与跟踪演示
python -m traffic_energy.detection.demo --source traffic_video.mp4

# 交通信号优化训练
python -m traffic_energy.signal_opt.train --algorithm PPO
```

#### 方向三：计算机节能（开发中）

```bash
# 启动进程监控
python -m computer_energy.monitor.daemon

# 手动执行节能检查
python -m computer_energy.scheduler.check_and_shutdown
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

详见 [docs/architecture/diagram.md](docs/architecture/diagram.md)

## 📚 开发指导文档

| 文档 | 内容 | 状态 |
|------|------|------|
| [项目需求文档](docs/project_requirements.md) | 三个方向的详细功能需求 | ✅ 已完成 |
| [方向一技术栈调研](docs/direction1_tech_stack_research.md) | 建筑节能技术选型与实现方案 | ✅ 已完成 |
| [方向二技术栈调研](docs/direction2_tech_stack_research.md) | 交通节能技术选型与实现方案 | ✅ 已完成 |
| [方向三技术栈调研](docs/direction3_tech_stack_research_computer.md) | 计算机节能技术选型与实现方案 | 🔄 待创建 |
| [基础设施复用分析](docs/infrastructure_reuse_analysis.md) | 三个方向可复用组件分析 | ✅ 已完成 |
| [亮度校准架构](docs/brightness_calibration_architecture.md) | 楼道灯控亮度校准设计 | ✅ 已完成 |
| [热控制架构](docs/thermal_control_v5.md) | 空调热控制架构设计 | ✅ 已完成 |

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
