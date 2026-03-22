#!/usr/bin/env python3
"""
路径分析模块

分析车辆行驶路径、转向比例和拥堵热点。
支持轨迹聚类和路径-时间图生成。

Example:
    >>> from traffic_energy.traffic_analysis import PathAnalyzer
    >>> analyzer = PathAnalyzer()
    >>> analyzer.add_trajectory(trajectory)
    >>> clusters = analyzer.cluster()
    >>> path_time_maps = analyzer.generate_path_time_map()
    >>> hotspots = analyzer.get_congestion_hotspots()
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import math

import numpy as np

from shared.logger import setup_logger

# 导入Module 3B相关模块
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
        CongestionLevel
    )
    MODULE_3B_AVAILABLE = True
except ImportError:
    MODULE_3B_AVAILABLE = False

logger = setup_logger("path_analyzer")


@dataclass
class PathSegment:
    """路径段
    
    Attributes:
        start_point: 起点
        end_point: 终点
        direction: 方向向量
        length: 长度
    """
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    direction: Tuple[float, float]
    length: float


@dataclass
class TurnEvent:
    """转向事件
    
    Attributes:
        track_id: 跟踪ID
        turn_type: 转向类型 ('left', 'right', 'straight', 'u_turn')
        angle: 转向角度
        location: 位置
    """
    track_id: int
    turn_type: str
    angle: float
    location: Tuple[float, float]


@dataclass
class CameraTopology:
    """摄像头拓扑结构
    
    定义摄像头位置和连接关系
    """
    camera_id: str
    position: Tuple[float, float]
    zones: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)


class PathAnalyzer:
    """路径分析器 (Module 3B)
    
    分析车辆行驶路径，支持轨迹聚类、路径-时间图生成和拥堵热点识别。
    
    Attributes:
        min_segment_length: 最小路径段长度
        turn_angle_threshold: 转向角度阈值
        clusterer: 轨迹聚类器
        flow_generator: 流量-时间矩阵生成器
        
    Example:
        >>> topology = CameraTopology("cam_001", (100, 100))
        >>> analyzer = PathAnalyzer(topology)
        >>> 
        >>> # 添加轨迹
        >>> for track in tracks:
        ...     trajectory = convert_track_to_trajectory(track)
        ...     analyzer.add_trajectory(trajectory)
        >>> 
        >>> # 执行聚类
        >>> clusters = analyzer.cluster()
        >>> 
        >>> # 生成路径-时间图
        >>> path_time_maps = analyzer.generate_path_time_map()
        >>> 
        >>> # 获取拥堵热点
        >>> hotspots = analyzer.get_congestion_hotspots(threshold="high")
    """
    
    def __init__(
        self,
        camera_topology: Optional[CameraTopology] = None,
        min_segment_length: float = 50.0,
        turn_angle_threshold: float = 30.0,
        cluster_eps: float = 50.0,
        cluster_min_samples: int = 5,
        time_window: int = 3600
    ) -> None:
        """初始化分析器
        
        Args:
            camera_topology: 摄像头拓扑结构
            min_segment_length: 最小路径段长度
            turn_angle_threshold: 转向角度阈值
            cluster_eps: 聚类半径
            cluster_min_samples: 最小聚类样本数
            time_window: 时间窗口 (秒)
        """
        self.topology = camera_topology
        self.min_segment_length = min_segment_length
        self.turn_angle_threshold = turn_angle_threshold
        
        # 原有数据结构
        self._trajectories: Dict[int, List] = {}
        self._turns: List[TurnEvent] = []
        self._entry_zones: Dict[str, int] = defaultdict(int)
        self._exit_zones: Dict[str, int] = defaultdict(int)
        
        # Module 3B 新增数据结构
        self._vehicle_trajectories: List[VehicleTrajectory] = []
        self._clusters: Dict[int, PathCluster] = {}
        
        if MODULE_3B_AVAILABLE:
            self.clusterer = TrajectoryClusterer(
                eps=cluster_eps,
                min_samples=cluster_min_samples
            )
            self.flow_generator = FlowTimeMatrixGenerator(time_window=time_window)
        else:
            self.clusterer = None
            self.flow_generator = None
            logger.warning("Module 3B 组件不可用，部分功能受限")
        
        logger.info("初始化路径分析器 (Module 3B)")
    
    def add_trajectory(
        self,
        track_id: int,
        trajectory: List,
        vehicle_type: str = "unknown",
        power_type: str = "unknown",
        entry_zone: Optional[str] = None,
        exit_zone: Optional[str] = None
    ) -> None:
        """添加轨迹
        
        Args:
            track_id: 跟踪ID
            trajectory: 轨迹点列表
            vehicle_type: 车辆类型
            power_type: 动力类型
            entry_zone: 进入区域
            exit_zone: 离开区域
        """
        if len(trajectory) < 3:
            return
        
        self._trajectories[track_id] = trajectory
        
        # 分析转向
        self._analyze_turns(track_id, trajectory)
        
        # 转换为VehicleTrajectory并存储 (Module 3B)
        if MODULE_3B_AVAILABLE:
            vehicle_traj = self._convert_to_vehicle_trajectory(
                track_id, trajectory, vehicle_type, power_type,
                entry_zone, exit_zone
            )
            if vehicle_traj:
                self._vehicle_trajectories.append(vehicle_traj)
    
    def add_vehicle_trajectory(
        self,
        trajectory: VehicleTrajectory
    ) -> None:
        """添加车辆轨迹对象 (Module 3B)
        
        Args:
            trajectory: VehicleTrajectory对象
        """
        if not MODULE_3B_AVAILABLE:
            logger.warning("Module 3B 组件不可用")
            return
        
        if trajectory.is_valid:
            self._vehicle_trajectories.append(trajectory)
    
    def _convert_to_vehicle_trajectory(
        self,
        track_id: int,
        trajectory_points: List,
        vehicle_type: str,
        power_type: str,
        entry_zone: Optional[str],
        exit_zone: Optional[str]
    ) -> Optional[VehicleTrajectory]:
        """将轨迹点列表转换为VehicleTrajectory
        
        Args:
            track_id: 跟踪ID
            trajectory_points: 轨迹点列表
            vehicle_type: 车辆类型
            power_type: 动力类型
            entry_zone: 进入区域
            exit_zone: 离开区域
            
        Returns:
            VehicleTrajectory对象或None
        """
        if not MODULE_3B_AVAILABLE or len(trajectory_points) < 3:
            return None
        
        # 转换轨迹点
        path_points = []
        entry_time = None
        exit_time = None
        
        for i, point in enumerate(trajectory_points):
            if hasattr(point, 'timestamp'):
                timestamp = point.timestamp
                if i == 0:
                    entry_time = datetime.fromtimestamp(timestamp)
                if i == len(trajectory_points) - 1:
                    exit_time = datetime.fromtimestamp(timestamp)
            else:
                timestamp = datetime.now().timestamp()
            
            center = point.center if hasattr(point, 'center') else point
            bbox = point.bbox if hasattr(point, 'bbox') else None
            
            path_points.append(TrajectoryPoint(
                timestamp=timestamp,
                center=center,
                bbox=bbox
            ))
        
        return VehicleTrajectory(
            track_id=track_id,
            vehicle_type=vehicle_type,
            power_type=power_type,
            entry_time=entry_time,
            exit_time=exit_time,
            entry_zone=entry_zone,
            exit_zone=exit_zone,
            path_points=path_points
        )
    
    def _analyze_turns(
        self,
        track_id: int,
        trajectory: List
    ) -> None:
        """分析转向
        
        Args:
            track_id: 跟踪ID
            trajectory: 轨迹点
        """
        points = [(p.center[0], p.center[1]) for p in trajectory if hasattr(p, 'center')]
        
        if len(points) < 3:
            return
        
        # 计算每点的方向
        for i in range(1, len(points) - 1):
            # 入方向
            dx1 = points[i][0] - points[i-1][0]
            dy1 = points[i][1] - points[i-1][1]
            
            # 出方向
            dx2 = points[i+1][0] - points[i][0]
            dy2 = points[i+1][1] - points[i][1]
            
            # 计算转向角度
            angle1 = math.atan2(dy1, dx1)
            angle2 = math.atan2(dy2, dx2)
            
            turn_angle = math.degrees(angle2 - angle1)
            
            # 归一化到 -180 ~ 180
            while turn_angle > 180:
                turn_angle -= 360
            while turn_angle < -180:
                turn_angle += 360
            
            # 判断转向类型
            if abs(turn_angle) < self.turn_angle_threshold:
                turn_type = 'straight'
            elif turn_angle > 0:
                if turn_angle > 150:
                    turn_type = 'u_turn'
                else:
                    turn_type = 'left'
            else:
                if turn_angle < -150:
                    turn_type = 'u_turn'
                else:
                    turn_type = 'right'
            
            if abs(turn_angle) >= self.turn_angle_threshold:
                turn_event = TurnEvent(
                    track_id=track_id,
                    turn_type=turn_type,
                    angle=turn_angle,
                    location=points[i]
                )
                self._turns.append(turn_event)
    
    def get_turn_ratio(self) -> Dict[str, float]:
        """获取转向比例
        
        Returns:
            {转向类型: 比例, ...}
        """
        if not self._turns:
            return {}
        
        turn_counts = defaultdict(int)
        for turn in self._turns:
            turn_counts[turn.turn_type] += 1
        
        total = len(self._turns)
        return {
            turn_type: count / total
            for turn_type, count in turn_counts.items()
        }
    
    def get_origin_destination_matrix(
        self,
        zones: Dict[str, List[Tuple[float, float]]]
    ) -> Dict[Tuple[str, str], int]:
        """获取OD矩阵
        
        Args:
            zones: 区域定义 {zone_id: [polygon_points]}
            
        Returns:
            {(origin, dest): count, ...}
        """
        od_matrix = defaultdict(int)
        
        for track_id, trajectory in self._trajectories.items():
            if len(trajectory) < 2:
                continue
            
            start_point = trajectory[0].center if hasattr(trajectory[0], 'center') else trajectory[0]
            end_point = trajectory[-1].center if hasattr(trajectory[-1], 'center') else trajectory[-1]
            
            origin = self._get_zone(start_point, zones)
            dest = self._get_zone(end_point, zones)
            
            if origin and dest:
                od_matrix[(origin, dest)] += 1
        
        return dict(od_matrix)
    
    def _get_zone(
        self,
        point: Tuple[float, float],
        zones: Dict[str, List[Tuple[float, float]]]
    ) -> Optional[str]:
        """获取点所属区域
        
        Args:
            point: 点坐标
            zones: 区域定义
            
        Returns:
            区域ID或None
        """
        import cv2
        
        for zone_id, polygon in zones.items():
            poly = np.array(polygon, dtype=np.int32)
            if cv2.pointPolygonTest(poly, point, False) >= 0:
                return zone_id
        
        return None
    
    def get_average_trajectory(self) -> Optional[List[Tuple[float, float]]]:
        """获取平均轨迹
        
        Returns:
            平均轨迹点列表或None
        """
        if not self._trajectories:
            return None
        
        # 简化：返回最长轨迹的中心线
        longest_trajectory = max(
            self._trajectories.values(),
            key=lambda t: len(t)
        )
        
        return [
            p.center if hasattr(p, 'center') else p
            for p in longest_trajectory
        ]
    
    def get_statistics(self) -> Dict:
        """获取统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'total_trajectories': len(self._trajectories),
            'total_turns': len(self._turns),
            'turn_ratio': self.get_turn_ratio(),
            'avg_trajectory_length': np.mean([
                len(t) for t in self._trajectories.values()
            ]) if self._trajectories else 0
        }
    
    def cluster(
        self,
        eps: Optional[float] = None,
        min_samples: Optional[int] = None
    ) -> Dict[int, PathCluster]:
        """执行轨迹聚类 (Module 3B)
        
        Args:
            eps: 聚类半径，None则使用初始化值
            min_samples: 最小样本数，None则使用初始化值
            
        Returns:
            聚类结果字典
        """
        if not MODULE_3B_AVAILABLE or self.clusterer is None:
            logger.warning("聚类功能不可用")
            return {}
        
        if not self._vehicle_trajectories:
            logger.warning("没有可用的轨迹数据")
            return {}
        
        # 更新参数
        if eps is not None:
            self.clusterer.eps = eps
        if min_samples is not None:
            self.clusterer.min_samples = min_samples
        
        # 执行聚类
        self._clusters = self.clusterer.cluster(self._vehicle_trajectories)
        
        logger.info(f"轨迹聚类完成，发现 {len(self._clusters)} 个路径模式")
        return self._clusters
    
    def generate_path_time_map(
        self,
        time_window: Optional[int] = None,
        reference_time: Optional[datetime] = None
    ) -> List[PathTimeMap]:
        """生成路径-时间图 (Module 3B)
        
        Args:
            time_window: 时间窗口 (秒)，None则使用初始化值
            reference_time: 参考时间
            
        Returns:
            路径-时间图列表
        """
        if not MODULE_3B_AVAILABLE or self.flow_generator is None:
            logger.warning("路径-时间图功能不可用")
            return []
        
        if not self._clusters:
            logger.warning("请先执行聚类")
            return []
        
        # 更新时间窗口
        if time_window is not None:
            self.flow_generator.time_window = time_window
        
        # 生成路径-时间图
        path_time_maps = self.flow_generator.generate(
            self._clusters,
            reference_time
        )
        
        return path_time_maps
    
    def get_congestion_hotspots(
        self,
        threshold: str = "medium",
        recent_only: bool = True
    ) -> List[PathCluster]:
        """获取拥堵热点 (Module 3B)
        
        Args:
            threshold: 拥堵阈值 ('low', 'medium', 'high')
            recent_only: 仅返回最近的数据
            
        Returns:
            拥堵热点路径聚类列表
        """
        if not MODULE_3B_AVAILABLE or self.flow_generator is None:
            logger.warning("拥堵热点功能不可用")
            return []
        
        # 获取拥堵的路径-时间图
        hotspots_maps = self.flow_generator.get_congestion_hotspots(
            threshold=threshold,
            recent_only=recent_only
        )
        
        # 转换为PathCluster
        hotspot_clusters = []
        for ptm in hotspots_maps:
            # 从path_id提取cluster_id
            try:
                cluster_id = int(ptm.path_id.replace("path_", ""))
                if cluster_id in self._clusters:
                    hotspot_clusters.append(self._clusters[cluster_id])
            except ValueError:
                continue
        
        return hotspot_clusters
    
    def get_cluster_statistics(self) -> Dict[str, Any]:
        """获取聚类统计信息 (Module 3B)
        
        Returns:
            统计信息字典
        """
        if not MODULE_3B_AVAILABLE or self.clusterer is None:
            return {"error": "Module 3B 不可用"}
        
        return self.clusterer.get_cluster_statistics()
    
    def get_path_statistics(
        self,
        path_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """获取路径统计信息 (Module 3B)
        
        Args:
            path_id: 路径ID
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            统计信息字典
        """
        if not MODULE_3B_AVAILABLE or self.flow_generator is None:
            return {"error": "Module 3B 不可用"}
        
        return self.flow_generator.get_path_statistics(path_id, start_time, end_time)
    
    def clear(self) -> None:
        """清除所有数据"""
        self._trajectories.clear()
        self._turns.clear()
        self._entry_zones.clear()
        self._exit_zones.clear()
        
        # Module 3B 数据清除
        self._vehicle_trajectories.clear()
        self._clusters.clear()
        
        if self.clusterer:
            self.clusterer.clear()
        if self.flow_generator:
            self.flow_generator.clear()
