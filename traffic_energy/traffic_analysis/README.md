# Module 3B: 路径分析系统

## 概述

Module 3B 是交通能源分析层的核心组件，专注于车辆轨迹聚类与路径-时间图生成。该系统使用DBSCAN算法分析YOLO跟踪数据，识别常见路径，并生成路径-时间图用于拥堵分析。

## 功能特性

- **轨迹聚类**: 使用DBSCAN算法对车辆轨迹进行聚类，识别常见行驶路径
- **路径-时间图**: 生成通行时间分布统计，支持多时间窗口分析
- **拥堵检测**: 基于通行时间比率计算拥堵程度
- **热点识别**: 自动识别拥堵热点路径

## 目录结构

```
traffic_energy/traffic_analysis/
├── __init__.py                 # 模块导出
├── path_analyzer.py            # 路径分析器（主入口）
├── trajectory_clustering.py    # 轨迹聚类模块
├── flow_time_matrix.py         # 流量-时间矩阵模块
├── congestion_detector.py      # 拥堵检测模块
├── flow_counter.py             # 流量计数器
└── tests/
    ├── __init__.py
    ├── test_clustering.py      # 聚类测试
    └── test_path_analyzer.py   # 路径分析器测试
```

## 核心类

### PathAnalyzer

路径分析器是Module 3B的主入口类，集成了轨迹聚类、路径-时间图生成和拥堵分析功能。

```python
from traffic_energy.traffic_analysis import PathAnalyzer, CameraTopology

# 创建摄像头拓扑
topology = CameraTopology(
    camera_id="cam_001",
    position=(320, 240),
    zones={
        "entry": [(0, 0), (100, 0), (100, 100), (0, 100)],
        "exit": [(540, 380), (640, 380), (640, 480), (540, 480)]
    }
)

# 初始化分析器
analyzer = PathAnalyzer(
    camera_topology=topology,
    cluster_eps=50.0,        # 聚类半径
    cluster_min_samples=5,   # 最小样本数
    time_window=3600         # 时间窗口（秒）
)

# 添加轨迹
analyzer.add_trajectory(
    track_id=1,
    trajectory=trajectory_points,
    vehicle_type="car",
    power_type="fuel"
)

# 执行聚类
clusters = analyzer.cluster()

# 生成路径-时间图
path_time_maps = analyzer.generate_path_time_map()

# 获取拥堵热点
hotspots = analyzer.get_congestion_hotspots(threshold="high")
```

### TrajectoryClusterer

轨迹聚类器使用DBSCAN算法对车辆轨迹进行聚类。

```python
from traffic_energy.traffic_analysis.trajectory_clustering import (
    TrajectoryClusterer,
    VehicleTrajectory,
    TrajectoryPoint
)

# 创建聚类器
clusterer = TrajectoryClusterer(eps=50.0, min_samples=5)

# 执行聚类
clusters = clusterer.cluster(trajectories)

# 获取聚类统计
stats = clusterer.get_cluster_statistics()
```

### FlowTimeMatrixGenerator

流量-时间矩阵生成器用于生成路径-时间图。

```python
from traffic_energy.traffic_analysis.flow_time_matrix import FlowTimeMatrixGenerator

# 创建生成器
generator = FlowTimeMatrixGenerator(time_window=3600)

# 生成路径-时间图
path_time_maps = generator.generate(clusters)

# 获取拥堵热点
hotspots = generator.get_congestion_hotspots(threshold="medium")
```

## 数据结构

### VehicleTrajectory

```python
@dataclass
class VehicleTrajectory:
    track_id: int              # 跟踪ID
    vehicle_type: str          # 车辆类型 (car/bus/truck)
    power_type: str           # 动力类型 (fuel/electric/unknown)
    entry_time: datetime      # 进入时间
    exit_time: datetime       # 离开时间
    entry_zone: str           # 进入区域
    exit_zone: str            # 离开区域
    path_points: List[TrajectoryPoint]  # 路径点列表
```

### PathCluster

```python
@dataclass
class PathCluster:
    cluster_id: int                    # 聚类ID
    representative_path: List[Tuple[float, float]]  # 代表性路径
    trajectories: List[VehicleTrajectory]           # 轨迹列表
    avg_travel_time: float             # 平均通行时间
    std_travel_time: float             # 通行时间标准差
    vehicle_type_dist: Dict[str, int]  # 车型分布
    power_type_dist: Dict[str, int]    # 油电分布
```

### PathTimeMap

```python
@dataclass
class PathTimeMap:
    timestamp: datetime       # 时间戳
    path_id: str             # 路径ID
    travel_times: List[float]  # 历史通行时间
    avg_time: float          # 平均通行时间
    min_time: float          # 最小通行时间
    max_time: float          # 最大通行时间
    congestion_level: str    # 拥堵等级 (low/medium/high)
    vehicle_count: int       # 车辆数量
```

## 拥堵等级定义

| 等级 | 阈值（通行时间/基准时间） | 描述 |
|------|------------------------|------|
| low | < 1.2 | 畅通 |
| medium | 1.2 - 1.5 | 轻度拥堵 |
| high | >= 1.5 | 严重拥堵 |

## 依赖

- numpy: 数值计算
- scikit-learn: DBSCAN聚类算法（可选）
- 现有VehicleTracker: 轨迹数据源

## 安装依赖

```bash
pip install numpy scikit-learn
```

## 运行测试

```bash
# 运行所有测试
python -m unittest discover traffic_energy/traffic_analysis/tests

# 运行特定测试
python -m unittest traffic_energy.traffic_analysis.tests.test_clustering
python -m unittest traffic_energy.traffic_analysis.tests.test_path_analyzer
```

## 性能指标

- **轨迹聚类准确率**: > 85%
- **路径识别**: 识别主要通行路径
- **拥堵检测延迟**: < 5分钟
- **数据处理速度**: 实时处理 (30fps)

## 接口说明

### 提供给 Module 3C (LLM拥堵识别) 的接口

```python
from traffic_energy.traffic_analysis.path_analyzer import PathAnalyzer
from traffic_energy.traffic_analysis.flow_time_matrix import PathTimeMap

# 使用示例
analyzer = PathAnalyzer(camera_topology)

# 在跟踪回调中添加轨迹
analyzer.add_trajectory(track_id, trajectory_points)

# 定期聚类
clusters = analyzer.cluster()

# 生成路径-时间图
path_time_maps = analyzer.generate_path_time_map()

# 获取拥堵热点
hotspots = analyzer.get_congestion_hotspots(threshold="high")
```

## 注意事项

1. **轨迹数据质量**: 跟踪可能丢失导致轨迹不完整，建议过滤短轨迹（< 3个点）
2. **DBSCAN参数**: eps和min_samples对结果影响大，建议根据实际场景调优
3. **时间同步**: 多摄像头时间可能不同步，建议实现时间校准
4. **车辆类型数据**: 依赖Module 3A的输出，需处理Module 3A未就绪的情况

## 版本历史

- v1.0 (2026-03-22): 初始版本，实现基本聚类和路径-时间图功能
