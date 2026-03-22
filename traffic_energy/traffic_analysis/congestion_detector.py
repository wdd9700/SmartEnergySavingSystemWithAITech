#!/usr/bin/env python3
"""
拥堵检测模块

基于速度、密度和流量进行拥堵检测。

Example:
    >>> from traffic_energy.traffic_analysis import CongestionDetector
    >>> detector = CongestionDetector()
    >>> level = detector.detect(tracks, road_length=100)
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time

import numpy as np

from shared.logger import setup_logger

logger = setup_logger("congestion_detector")


class CongestionLevel(Enum):
    """拥堵等级"""
    FREE_FLOW = "free_flow"          # 畅通
    LIGHT = "light_congestion"       # 轻度拥堵
    MODERATE = "moderate_congestion" # 中度拥堵
    SEVERE = "severe_congestion"     # 严重拥堵


@dataclass
class CongestionStatus:
    """拥堵状态
    
    Attributes:
        level: 拥堵等级
        density: 车辆密度 (辆/km)
        avg_speed: 平均速度 (km/h)
        occupancy: 占有率 (0-1)
        confidence: 置信度
        timestamp: 时间戳
    """
    level: CongestionLevel
    density: float
    avg_speed: float
    occupancy: float
    confidence: float
    timestamp: float


class CongestionDetector:
    """拥堵检测器
    
    基于交通流理论进行拥堵检测。
    
    Attributes:
        speed_threshold_slow: 低速阈值
        speed_threshold_congested: 拥堵速度阈值
        density_threshold: 密度阈值
        
    Example:
        >>> detector = CongestionDetector(
        ...     speed_threshold_slow=20,
        ...     speed_threshold_congested=10
        ... )
        >>> status = detector.detect(tracks, road_length=100, road_width=10)
    """
    
    def __init__(
        self,
        speed_threshold_slow: float = 20.0,      # km/h
        speed_threshold_congested: float = 10.0, # km/h
        density_threshold: float = 0.7,
        min_vehicles: int = 5
    ) -> None:
        """初始化拥堵检测器
        
        Args:
            speed_threshold_slow: 低速阈值 (km/h)
            speed_threshold_congested: 拥堵速度阈值 (km/h)
            density_threshold: 密度阈值
            min_vehicles: 最小车辆数
        """
        self.speed_threshold_slow = speed_threshold_slow
        self.speed_threshold_congested = speed_threshold_congested
        self.density_threshold = density_threshold
        self.min_vehicles = min_vehicles
        
        self._history: List[CongestionStatus] = []
        
        logger.info("初始化拥堵检测器")
    
    def detect(
        self,
        tracks: List,
        road_length: float,  # 米
        road_width: float = 10.0,  # 米
        speed_measurements: Optional[Dict[int, float]] = None
    ) -> CongestionStatus:
        """检测拥堵状态
        
        Args:
            tracks: 跟踪轨迹列表
            road_length: 道路长度（米）
            road_width: 道路宽度（米）
            speed_measurements: 速度测量值 {track_id: speed_kmh}
            
        Returns:
            拥堵状态
        """
        # 过滤有效轨迹
        valid_tracks = [t for t in tracks if t.is_confirmed]
        
        if len(valid_tracks) < self.min_vehicles:
            return CongestionStatus(
                level=CongestionLevel.FREE_FLOW,
                density=0.0,
                avg_speed=0.0,
                occupancy=0.0,
                confidence=1.0,
                timestamp=time.time()
            )
        
        # 计算密度 (辆/km)
        road_length_km = road_length / 1000.0
        density = len(valid_tracks) / road_length_km if road_length_km > 0 else 0
        
        # 计算平均速度
        if speed_measurements:
            speeds = [
                speed_measurements.get(t.track_id, 0)
                for t in valid_tracks
                if t.track_id in speed_measurements
            ]
            avg_speed = np.mean(speeds) if speeds else 0
        else:
            # 从轨迹估计速度
            speeds = []
            for track in valid_tracks:
                if track.trajectory and len(track.trajectory) >= 2:
                    # 简单速度估计
                    speed = self._estimate_speed_from_trajectory(track)
                    if speed > 0:
                        speeds.append(speed)
            avg_speed = np.mean(speeds) if speeds else 30.0
        
        # 计算占有率
        road_area = road_length * road_width
        vehicle_area = sum(
            (t.bbox[2] - t.bbox[0]) * (t.bbox[3] - t.bbox[1])
            for t in valid_tracks
        )
        occupancy = min(1.0, vehicle_area / road_area) if road_area > 0 else 0
        
        # 判断拥堵等级
        level = self._classify_congestion(avg_speed, density, occupancy)
        
        # 计算置信度
        confidence = self._calculate_confidence(
            len(valid_tracks), avg_speed, density
        )
        
        status = CongestionStatus(
            level=level,
            density=density,
            avg_speed=avg_speed,
            occupancy=occupancy,
            confidence=confidence,
            timestamp=time.time()
        )
        
        self._history.append(status)
        
        # 限制历史长度
        if len(self._history) > 1000:
            self._history = self._history[-500:]
        
        return status
    
    def _estimate_speed_from_trajectory(self, track) -> float:
        """从轨迹估计速度
        
        Args:
            track: 轨迹对象
            
        Returns:
            估计速度 (km/h)
        """
        if len(track.trajectory) < 2:
            return 0.0
        
        # 获取最近几个点
        points = track.trajectory[-5:]
        
        if len(points) < 2:
            return 0.0
        
        # 计算平均速度
        total_distance = 0.0
        total_time = 0.0
        
        for i in range(1, len(points)):
            p1 = points[i - 1]
            p2 = points[i]
            
            dx = p2.center[0] - p1.center[0]
            dy = p2.center[1] - p1.center[1]
            distance = np.sqrt(dx**2 + dy**2)
            dt = p2.timestamp - p1.timestamp
            
            if dt > 0:
                total_distance += distance
                total_time += dt
        
        if total_time > 0:
            # 假设像素到米的转换比例为 0.1
            pixels_per_meter = 0.1
            speed_ms = total_distance * pixels_per_meter / total_time
            speed_kmh = speed_ms * 3.6
            return speed_kmh
        
        return 0.0
    
    def _classify_congestion(
        self,
        avg_speed: float,
        density: float,
        occupancy: float
    ) -> CongestionLevel:
        """分类拥堵等级
        
        Args:
            avg_speed: 平均速度
            density: 密度
            occupancy: 占有率
            
        Returns:
            拥堵等级
        """
        # 严重拥堵
        if avg_speed < self.speed_threshold_congested or occupancy > 0.9:
            return CongestionLevel.SEVERE
        
        # 中度拥堵
        if avg_speed < self.speed_threshold_slow or occupancy > self.density_threshold:
            return CongestionLevel.MODERATE
        
        # 轻度拥堵
        if avg_speed < self.speed_threshold_slow * 1.5 or occupancy > 0.4:
            return CongestionLevel.LIGHT
        
        # 畅通
        return CongestionLevel.FREE_FLOW
    
    def _calculate_confidence(
        self,
        num_vehicles: int,
        avg_speed: float,
        density: float
    ) -> float:
        """计算检测置信度
        
        Args:
            num_vehicles: 车辆数量
            avg_speed: 平均速度
            density: 密度
            
        Returns:
            置信度 (0-1)
        """
        # 基于样本量的置信度
        sample_confidence = min(1.0, num_vehicles / self.min_vehicles)
        
        # 基于速度稳定性的置信度
        speed_confidence = 1.0 if avg_speed > 0 else 0.5
        
        # 综合置信度
        confidence = (sample_confidence + speed_confidence) / 2.0
        
        return confidence
    
    def get_trend(self, window_size: int = 10) -> str:
        """获取拥堵趋势
        
        Args:
            window_size: 窗口大小
            
        Returns:
            趋势描述 ('improving', 'worsening', 'stable')
        """
        if len(self._history) < window_size * 2:
            return "unknown"
        
        recent = self._history[-window_size:]
        previous = self._history[-window_size*2:-window_size]
        
        recent_level = sum(r.level.value != "free_flow" for r in recent) / len(recent)
        previous_level = sum(p.level.value != "free_flow" for p in previous) / len(previous)
        
        if recent_level < previous_level - 0.1:
            return "improving"
        elif recent_level > previous_level + 0.1:
            return "worsening"
        else:
            return "stable"
    
    def get_statistics(self) -> Dict:
        """获取统计信息
        
        Returns:
            统计信息字典
        """
        if not self._history:
            return {}
        
        levels = [s.level for s in self._history]
        
        return {
            'total_samples': len(self._history),
            'level_distribution': {
                level.value: levels.count(level)
                for level in CongestionLevel
            },
            'avg_density': np.mean([s.density for s in self._history]),
            'avg_speed': np.mean([s.avg_speed for s in self._history]),
            'current_trend': self.get_trend()
        }
    
    def detect_from_path_time_map(
        self,
        current_time: float,
        baseline_time: float,
        vehicle_count: int
    ) -> CongestionLevel:
        """基于路径-时间图检测拥堵 (Module 3B)
        
        使用通行时间比率来判断拥堵程度
        
        Args:
            current_time: 当前平均通行时间 (秒)
            baseline_time: 基准通行时间 (秒)
            vehicle_count: 车辆数量
            
        Returns:
            拥堵等级
        """
        if baseline_time <= 0 or vehicle_count < self.min_vehicles:
            return CongestionLevel.FREE_FLOW
        
        ratio = current_time / baseline_time
        
        # 严重拥堵: 通行时间超过基准1.5倍
        if ratio >= 1.5:
            return CongestionLevel.SEVERE
        
        # 中度拥堵: 通行时间超过基准1.2倍
        if ratio >= 1.2:
            return CongestionLevel.MODERATE
        
        # 轻度拥堵: 通行时间超过基准1.1倍
        if ratio >= 1.1:
            return CongestionLevel.LIGHT
        
        return CongestionLevel.FREE_FLOW
    
    def calculate_congestion_from_travel_time(
        self,
        travel_time: float,
        free_flow_time: float
    ) -> Tuple[str, float]:
        """根据通行时间计算拥堵等级和拥堵指数 (Module 3B)
        
        Args:
            travel_time: 实际通行时间
            free_flow_time: 自由流通行时间
            
        Returns:
            (拥堵等级, 拥堵指数)
        """
        if free_flow_time <= 0:
            return "unknown", 0.0
        
        ratio = travel_time / free_flow_time
        congestion_index = max(0.0, ratio - 1.0)  # 拥堵指数
        
        if ratio < 1.2:
            level = "low"
        elif ratio < 1.5:
            level = "medium"
        else:
            level = "high"
        
        return level, congestion_index
    
    def identify_congestion_hotspots(
        self,
        path_time_data: List[Dict],
        threshold_ratio: float = 1.3
    ) -> List[Dict]:
        """识别拥堵热点 (Module 3B)
        
        Args:
            path_time_data: 路径-时间数据列表
                [{"path_id": str, "current_time": float, "baseline_time": float}, ...]
            threshold_ratio: 拥堵阈值比率
            
        Returns:
            拥堵热点列表
        """
        hotspots = []
        
        for data in path_time_data:
            path_id = data.get("path_id")
            current_time = data.get("current_time", 0)
            baseline_time = data.get("baseline_time", 0)
            
            if baseline_time <= 0:
                continue
            
            ratio = current_time / baseline_time
            
            if ratio >= threshold_ratio:
                level, index = self.calculate_congestion_from_travel_time(
                    current_time, baseline_time
                )
                
                hotspots.append({
                    "path_id": path_id,
                    "congestion_level": level,
                    "congestion_index": index,
                    "current_time": current_time,
                    "baseline_time": baseline_time,
                    "delay_ratio": ratio
                })
        
        # 按拥堵指数排序
        hotspots.sort(key=lambda x: x["congestion_index"], reverse=True)
        
        return hotspots
