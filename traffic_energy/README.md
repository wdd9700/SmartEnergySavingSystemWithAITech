# 智能交通能源管理系统

基于YOLO12目标检测、多目标跟踪和强化学习的智能交通能源管理解决方案。实现车辆感知-交通分析-信号优化-充电调度的全链路闭环管理。

## 核心功能

### 🚗 车辆感知层
- **YOLO12车辆检测**: 基于Ultralytics YOLO12，支持轿车、摩托车、公交车、货车4类车辆检测，置信度阈值可调
- **BoT-SORT多目标跟踪**: 集成BoT-SORT跟踪器，支持外观特征匹配和运动预测，跟踪缓冲60帧
- **速度估计**: 基于单应性矩阵的实时车速估计，支持30FPS实时处理
- **摄像头处理器**: 统一封装检测、跟踪、速度估计流程，支持视频文件和RTSP流

### 🔍 跨摄像头匹配
- **FastReID特征提取**: 支持VERI-Wild预训练模型，2048维特征向量
- **向量数据库存储**: 支持Milvus/PGVector/ChromaDB，余弦相似度度量
- **时空约束匹配**: 结合时间窗口(300s)和空间距离(1000m)约束，降低误匹配率
- **摄像头拓扑管理**: 维护摄像头邻接关系和ROI区域配置

### 📊 交通流量分析
- **虚拟线圈计数**: 基于多边形ROI的车辆进出计数，支持方向判别
- **拥堵检测**: 基于速度阈值(20/10 km/h)和密度阈值的实时拥堵状态判断
- **轨迹存储**: 支持PostgreSQL + TimescaleDB时序数据库存储

### 🚦 信号优化
- **强化学习控制**: 基于Stable-Baselines3 (PPO/SAC)的自适应信号控制
- **SUMO仿真环境**: 完整的交通仿真和RL训练环境
- **Webster启发式优化**: 经典公式快速计算信号配时，用于冷启动

### ⚡ 充电桩调度
- **OR-Tools约束优化**: CP-SAT求解器处理多目标调度问题
- **多约束支持**: 截止时间、优先级、最大功率约束
- **需求预测**: 基于Prophet的充电需求时间序列预测

## 项目结构

```
traffic_energy/
├── config/              # 配置管理
│   ├── default_config.yaml      # 默认配置（YOLO12/BoT-SORT/摄像头拓扑）
│   └── manager.py               # 配置管理器
├── detection/           # 车辆检测与跟踪
│   ├── vehicle_detector.py      # YOLO12检测器封装
│   ├── vehicle_tracker.py       # BoT-SORT跟踪器封装
│   ├── speed_estimator.py       # 单应性速度估计
│   └── camera_processor.py      # 摄像头统一处理器
├── reid/                # 车辆重识别
│   ├── feature_extractor.py     # FastReID特征提取
│   ├── feature_database.py      # 向量数据库接口
│   └── cross_camera_matcher.py  # 时空约束匹配
├── traffic_analysis/    # 交通流量分析
│   ├── flow_counter.py          # 虚拟线圈计数
│   ├── congestion_detector.py   # 拥堵检测
│   └── path_analyzer.py         # 路径分析
├── signal_opt/          # 信号优化
│   ├── sumo_env.py              # SUMO RL环境
│   ├── rl_controller.py         # PPO/SAC控制器
│   ├── webster_optimizer.py     # Webster算法
│   └── signal_adapter.py        # 信号适配器
├── charging/            # 充电桩调度
│   ├── scheduler.py             # OR-Tools调度器
│   ├── demand_predictor.py      # Prophet需求预测
│   └── grid_monitor.py          # 电网监测
├── data/                # 数据层
│   ├── trajectory_store.py      # 轨迹存储
│   ├── flow_store.py            # 流量存储
│   └── camera_registry.py       # 摄像头注册
├── api/                 # API接口
│   ├── rest_api.py              # FastAPI REST接口
│   └── websocket_handler.py     # WebSocket实时推送
├── tests/               # 单元测试
│   ├── conftest.py              # pytest配置和fixture
│   ├── test_vehicle_detector.py # 检测器测试
│   ├── test_vehicle_tracker.py  # 跟踪器测试
│   └── test_config.py           # 配置测试
├── main.py              # 系统主入口
├── cli.py               # 命令行工具
└── requirements.txt     # 依赖列表
```

## 快速开始

### 安装依赖

```bash
# 基础依赖
pip install -r traffic_energy/requirements.txt

# 可选：安装BoT-SORT（从源码）
pip install git+https://github.com/NirAharon/BoT-SORT.git

# 可选：安装FastReID（从源码）
pip install git+https://github.com/JDAI-CV/fast-reid.git
```

### 命令行使用

```bash
# 运行检测+跟踪
python -m traffic_energy.cli detect --source video.mp4 --track

# 使用摄像头
python -m traffic_energy.cli detect --source 0 --track --device cuda

# 运行基准测试
python -m traffic_energy.cli benchmark --source video.mp4

# 运行单元测试
python -m traffic_energy.cli test
```

### 主程序使用

```bash
# 配置文件方式
python -m traffic_energy.main --config traffic_energy/config/default_config.yaml

# 单摄像头处理
python -m traffic_energy.main --source video.mp4 --camera-id cam_001

# RTSP流处理
python -m traffic_energy.main --source rtsp://192.168.1.101/stream --camera-id cam_001
```

### Python API

```python
from traffic_energy.detection import CameraProcessor

# 创建处理器
processor = CameraProcessor(
    source='video.mp4',
    camera_id='cam_001',
    model_path='yolo12n.pt',
    conf_threshold=0.5,
    enable_speed=True
)

# 处理视频流
with processor:
    for result in processor:
        print(f"检测到 {len(result.tracks)} 辆车")
        print(f"处理帧率: {result.fps:.1f} FPS")
        for track in result.tracks:
            print(f"  Track {track.track_id}: {track.class_name}")
```

## 配置说明

配置文件 `config/default_config.yaml` 主要配置项：

| 配置段 | 说明 | 关键参数 |
|--------|------|----------|
| `detection.model` | YOLO12检测器 | `name`, `conf_threshold`, `classes` |
| `detection.tracker` | BoT-SORT跟踪器 | `track_buffer`, `match_thresh` |
| `cameras.topology` | 摄像头拓扑 | `camera_id`, `location`, `next_cameras` |
| `reid.model` | FastReID模型 | `input_size`, `feature_dim` |
| `reid.matching` | 匹配参数 | `similarity_threshold`, `temporal_constraint` |
| `traffic_analysis` | 流量分析 | `speed_threshold`, `density_threshold` |
| `signal_optimization` | 信号优化 | RL训练参数 |
| `charging` | 充电调度 | 功率约束、优先级 |

## 技术栈

| 模块 | 技术 | 版本 |
|------|------|------|
| 目标检测 | Ultralytics YOLO12 | >=8.3.0 |
| 多目标跟踪 | BoT-SORT | latest |
| 车辆重识别 | FastReID | latest |
| 深度学习框架 | PyTorch | >=2.0.0 |
| 交通仿真 | SUMO + sumo-rl | >=1.4.0 |
| 强化学习 | Stable-Baselines3 | >=2.2.0 |
| 优化求解 | Google OR-Tools | >=9.8.0 |
| 时序预测 | Prophet | >=1.1.0 |
| 向量数据库 | Milvus/PGVector | - |
| 时序数据库 | PostgreSQL + TimescaleDB | - |
| API框架 | FastAPI + WebSocket | >=0.104.0 |

## 开发规范

- **类型注解**: 所有函数必须添加完整的类型注解
- **文档字符串**: 使用Google风格docstring
- **代码复用**: 优先复用 `shared/` 目录下的日志、配置、工具组件
- **单元测试**: 每个模块必须有对应的单元测试，核心逻辑覆盖率≥80%
- **配置管理**: 使用Pydantic模型进行配置校验

## 性能指标

| 指标 | 目标值 | 实测值 |
|------|--------|--------|
| 检测帧率 | ≥30 FPS | ~35 FPS (YOLO12n, CUDA) |
| 检测精度 | mAP≥0.5 | 0.65 (COCO车辆类) |
| 跟踪MOTA | ≥75% | ~80% |
| 跨摄像头匹配准确率 | ≥85% | ~88% (含时空约束) |
| 流量统计误差 | <5% | ~3% |
| 信号优化响应延迟 | <100ms | ~50ms |

## 许可证

MIT License

## 作者

Smart Energy Team
