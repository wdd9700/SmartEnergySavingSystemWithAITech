#!/usr/bin/env python3
"""
流量-时间矩阵模块

生成路径-时间图，用于拥堵分析和交通流量统计。

Example:
    >>> from traffic_energy.traffic_analysis.flow_time_matrix import FlowTimeMatrixGenerator, PathTimeMap
    >>> generator = FlowTimeMatrixGenerator(time_window=3600)
    >>> path_time_maps = generator.generate(clusters)
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum

import numpy as np

from shared.logger import setup_logger

logger = setup_logger("flow_time_matrix")


class CongestionLevel(Enum):
    """拥堵等级枚举"""
    LOW = "low"           # 畅通
    MEDIUM = "medium"     # 轻度拥堵
    HIGH = "high"         # 严重拥堵


@dataclass
class PathTimeMap:
    """路径-时间图数据类
    
    Attributes:
        timestamp: 时间戳
        path_id: 路径ID
        travel_times: 历史通行时间列表
        avg_time: 平均通行时间
        min_time: 最小通行时间
        max_time: 最大通行时间
        congestion_level: 拥堵等级
        vehicle_count: 车辆数量
    """
    timestamp: datetime
    path_id: str
    travel_times: List[float] = field(default_factory=list)
    avg_time: float = 0.0
    min_time: float = 0.0
    max_time: float = 0.0
    congestion_level: str = "low"
    vehicle_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "path_id": self.path_id,
            "avg_time": self.avg_time,
            "min_time": self.min_time,
            "max_time": self.max_time,
            "congestion_level": self.congestion_level,
            "vehicle_count": self.vehicle_count
        }


@dataclass
class FlowTimeEntry:
    """流量-时间条目
    
    用于存储特定时间段内的流量和通行时间数据
    """
    start_time: datetime
    end_time: datetime
    vehicle_count: int = 0
    travel_times: List[float] = field(default_factory=list)
    
    @property
    def avg_travel_time(self) -> float:
        """平均通行时间"""
        return np.mean(self.travel_times) if self.travel_times else 0.0
    
    @property
    def flow_rate(self) -> float:
        """流量 (辆/小时)"""
        duration_hours = (self.end_time - self.start_time).total_seconds() / 3600
        return self.vehicle_count / duration_hours if duration_hours > 0 else 0.0


class FlowTimeMatrixGenerator:
    """流量-时间矩阵生成器
    
    生成路径-时间图，用于分析交通流量和拥堵状况。
    
    Attributes:
        time_window: 时间窗口 (秒)
        baseline_times: 各路径的基准通行时间
        
    Example:
        >>> generator = FlowTimeMatrixGenerator(time_window=3600)  # 1小时窗口
        >>> path_time_maps = generator.generate(clusters)
        >>> for ptm in path_time_maps:
        ...     print(f"{ptm.path_id}: {ptm.congestion_level}")
    """
    
    def __init__(
        self,
        time_window: int = 3600,
        low_congestion_threshold: float = 1.2,
        high_congestion_threshold: float = 1.5
    ) -> None:
        """初始化流量-时间矩阵生成器
        
        Args:
            time_window: 时间窗口 (秒)，默认3600秒(1小时)
            low_congestion_threshold: 轻度拥堵阈值 (当前时间/基准时间)
            high_congestion_threshold: 严重拥堵阈值
        """
        self.time_window = time_window
        self.low_congestion_threshold = low_congestion_threshold
        self.high_congestion_threshold = high_congestion_threshold
        
        self._baseline_times: Dict[str, float] = {}
        self._path_time_maps: List[PathTimeMap] = []
        
        logger.info(f"初始化流量-时间矩阵生成器 (time_window={time_window}s)")
    
    def generate(
        self,
        clusters: Dict[int, Any],
        reference_time: Optional[datetime] = None
    ) -> List[PathTimeMap]:
        """生成路径-时间图
        
        Args:
            clusters: 路径聚类结果 {cluster_id: PathCluster}
            reference_time: 参考时间，默认为当前时间
            
        Returns:
            路径-时间图列表
        """
        if reference_time is None:
            reference_time = datetime.now()
        
        path_time_maps = []
        
        for cluster_id, cluster in clusters.items():
            path_id = f"path_{cluster_id}"
            
            # 更新基准时间
            if cluster.avg_travel_time > 0:
                self._baseline_times[path_id] = cluster.avg_travel_time
            
            # 按时间窗口分组
            time_groups = self._group_by_time(
                cluster.trajectories,
                reference_time
            )
            
            for timestamp, trajectories in time_groups.items():
                travel_times = [t.travel_time for t in trajectories if t.travel_time > 0]
                
                if not travel_times:
                    continue
                
                avg_time = np.mean(travel_times)
                
                # 计算拥堵程度
                baseline = self._baseline_times.get(path_id, avg_time)
                congestion = self._calculate_congestion(avg_time, baseline)
                
                path_time_map = PathTimeMap(
                    timestamp=timestamp,
                    path_id=path_id,
                    travel_times=travel_times,
                    avg_time=avg_time,
                    min_time=np.min(travel_times),
                    max_time=np.max(travel_times),
                    congestion_level=congestion,
                    vehicle_count=len(trajectories)
                )
                
                path_time_maps.append(path_time_map)
        
        self._path_time_maps.extend(path_time_maps)
        
        logger.info(f"生成了 {len(path_time_maps)} 个路径-时间图")
        return path_time_maps
    
    def _group_by_time(
        self,
        trajectories: List[Any],
        reference_time: datetime
    ) -> Dict[datetime, List[Any]]:
        """按时间窗口对轨迹进行分组
        
        Args:
            trajectories: 轨迹列表
            reference_time: 参考时间
            
        Returns:
            按时间分组的轨迹字典
        """
        groups: Dict[datetime, List[Any]] = defaultdict(list)
        
        window_delta = timedelta(seconds=self.time_window)
        
        for traj in trajectories:
            if traj.entry_time is None:
                continue
            
            # 计算所属时间窗口
            time_diff = reference_time - traj.entry_time
            window_index = int(time_diff.total_seconds() / self.time_window)
            
            # 窗口时间戳（取窗口开始时间）
            window_time = reference_time - timedelta(
                seconds=window_index * self.time_window
            )
            
            groups[window_time].append(traj)
        
        return dict(groups)
    
    def _calculate_congestion(
        self,
        current_time: float,
        baseline_time: float
    ) -> str:
        """计算拥堵程度
        
        Args:
            current_time: 当前平均通行时间
            baseline_time: 基准通行时间
            
        Returns:
            拥堵等级 ('low', 'medium', 'high')
        """
        if baseline_time <= 0:
            return CongestionLevel.LOW.value
        
        ratio = current_time / baseline_time
        
        if ratio < self.low_congestion_threshold:
            return CongestionLevel.LOW.value
        elif ratio < self.high_congestion_threshold:
            return CongestionLevel.MEDIUM.value
        else:
            return CongestionLevel.HIGH.value
    
    def get_congestion_hotspots(
        self,
        threshold: str = "medium",
        recent_only: bool = True,
        window_count: int = 1
    ) -> List[PathTimeMap]:
        """获取拥堵热点
        
        Args:
            threshold: 拥堵阈值 ('low', 'medium', 'high')
            recent_only: 仅返回最近的数据
            window_count: 最近窗口数量
            
        Returns:
            拥堵热点列表
        """
        if not self._path_time_maps:
            return []
        
        # 筛选符合条件的
        hotspots = [
            ptm for ptm in self._path_time_maps
            if self._congestion_level_compare(ptm.congestion_level, threshold)
        ]
        
        if recent_only and hotspots:
            # 按路径分组，取最近的数据
            path_groups: Dict[str, List[PathTimeMap]] = defaultdict(list)
            for ptm in hotspots:
                path_groups[ptm.path_id].append(ptm)
            
            # 对每个路径取最近的N个
            recent_hotspots = []
            for path_id, maps in path_groups.items():
                sorted_maps = sorted(maps, key=lambda x: x.timestamp, reverse=True)
                recent_hotspots.extend(sorted_maps[:window_count])
            
            return recent_hotspots
        
        return hotspots
    
    def _congestion_level_compare(
        self,
        level: str,
        threshold: str
    ) -> bool:
        """比较拥堵等级
        
        Args:
            level: 当前拥堵等级
            threshold: 阈值等级
            
        Returns:
            是否达到或超过阈值
        """
        levels = ["low", "medium", "high"]
        
        try:
            level_idx = levels.index(level)
            threshold_idx = levels.index(threshold)
            return level_idx >= threshold_idx
        except ValueError:
            return False
    
    def get_flow_matrix(
        self,
        path_ids: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, List[FlowTimeEntry]]:
        """获取流量-时间矩阵
        
        Args:
            path_ids: 路径ID列表，None表示所有路径
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            流量-时间矩阵 {path_id: [FlowTimeEntry, ...]}
        """
        # 筛选数据
        filtered_maps = self._path_time_maps
        
        if path_ids:
            filtered_maps = [m for m in filtered_maps if m.path_id in path_ids]
        
        if start_time:
            filtered_maps = [m for m in filtered_maps if m.timestamp >= start_time]
        
        if end_time:
            filtered_maps = [m for m in filtered_maps if m.timestamp <= end_time]
        
        # 按路径分组
        path_groups: Dict[str, List[PathTimeMap]] = defaultdict(list)
        for ptm in filtered_maps:
            path_groups[ptm.path_id].append(ptm)
        
        # 转换为FlowTimeEntry
        result = {}
        for path_id, maps in path_groups.items():
            entries = []
            for m in sorted(maps, key=lambda x: x.timestamp):
                entry = FlowTimeEntry(
                    start_time=m.timestamp,
                    end_time=m.timestamp + timedelta(seconds=self.time_window),
                    vehicle_count=m.vehicle_count,
                    travel_times=m.travel_times
                )
                entries.append(entry)
            result[path_id] = entries
        
        return result
    
    def get_path_statistics(
        self,
        path_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """获取路径统计信息
        
        Args:
            path_id: 路径ID
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            统计信息字典
        """
        # 筛选数据
        maps = [m for m in self._path_time_maps if m.path_id == path_id]
        
        if start_time:
            maps = [m for m in maps if m.timestamp >= start_time]
        
        if end_time:
            maps = [m for m in maps if m.timestamp <= end_time]
        
        if not maps:
            return {
                "path_id": path_id,
                "data_points": 0,
                "avg_travel_time": 0.0,
                "total_vehicles": 0
            }
        
        all_times = []
        total_vehicles = 0
        congestion_counts = defaultdict(int)
        
        for m in maps:
            all_times.extend(m.travel_times)
            total_vehicles += m.vehicle_count
            congestion_counts[m.congestion_level] += 1
        
        return {
            "path_id": path_id,
            "data_points": len(maps),
            "avg_travel_time": np.mean(all_times) if all_times else 0.0,
            "std_travel_time": np.std(all_times) if all_times else 0.0,
            "total_vehicles": total_vehicles,
            "avg_vehicles_per_window": total_vehicles / len(maps),
            "congestion_distribution": dict(congestion_counts)
        }
    
    def get_baseline_time(self, path_id: str) -> float:
        """获取路径的基准通行时间
        
        Args:
            path_id: 路径ID
            
        Returns:
            基准通行时间
        """
        return self._baseline_times.get(path_id, 0.0)
    
    def set_baseline_time(self, path_id: str, baseline: float) -> None:
        """设置路径的基准通行时间
        
        Args:
            path_id: 路径ID
            baseline: 基准通行时间
        """
        self._baseline_times[path_id] = baseline
        logger.info(f"设置路径 {path_id} 的基准通行时间为 {baseline:.2f}秒")
    
    def clear(self) -> None:
        """清除所有数据"""
        self._baseline_times.clear()
        self._path_time_maps.clear()
        logger.info("流量-时间矩阵数据已清除")
    
    def get_all_path_ids(self) -> List[str]:
        """获取所有路径ID
        
        Returns:
            路径ID列表
        """
        return list(set(m.path_id for m in self._path_time_maps))
