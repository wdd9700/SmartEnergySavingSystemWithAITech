#!/usr/bin/env python3
"""交通流量分析模块 (Module 3B)

路径分析系统 - 车辆轨迹聚类与路径-时间图生成
"""

from .flow_counter import FlowCounter
from .path_analyzer import PathAnalyzer, CameraTopology
from .congestion_detector import CongestionDetector, CongestionLevel, CongestionStatus

# Module 3B 新增导出
try:
    from .trajectory_clustering import (
        TrajectoryClusterer,
        VehicleTrajectory,
        PathCluster,
        TrajectoryPoint
    )
    from .flow_time_matrix import (
        FlowTimeMatrixGenerator,
        PathTimeMap,
        FlowTimeEntry,
        CongestionLevel as PathCongestionLevel
    )
    MODULE_3B_AVAILABLE = True
except ImportError:
    MODULE_3B_AVAILABLE = False

__all__ = [
    'FlowCounter',
    'PathAnalyzer',
    'CameraTopology',
    'CongestionDetector',
    'CongestionLevel',
    'CongestionStatus'
]

# 添加Module 3B导出
if MODULE_3B_AVAILABLE:
    __all__.extend([
        'TrajectoryClusterer',
        'VehicleTrajectory',
        'PathCluster',
        'TrajectoryPoint',
        'FlowTimeMatrixGenerator',
        'PathTimeMap',
        'FlowTimeEntry',
        'PathCongestionLevel'
    ])
