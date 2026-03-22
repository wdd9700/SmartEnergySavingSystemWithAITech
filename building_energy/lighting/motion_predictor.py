#!/usr/bin/env python3
"""
运动预测器

实现人员运动方向预测和到达时间估算，支持预测性照明控制。

主要功能:
- 运动轨迹跟踪
- 方向预测
- 到达时间估算
- 目标区域预测
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import math


@dataclass
class MotionEvent:
    """
    运动事件数据
    
    Attributes:
        timestamp: 事件时间戳
        zone_id: 当前区域ID
        position: 位置坐标 (x, y)
        direction: 运动方向 (角度，0-360度，0为东)
        speed: 运动速度 (m/s)
        confidence: 检测置信度 (0-1)
        track_id: 跟踪ID (用于关联同一人的连续事件)
    """
    timestamp: datetime
    zone_id: str
    position: Tuple[int, int]
    direction: float
    speed: float
    confidence: float
    track_id: Optional[str] = None


@dataclass
class ZoneConfig:
    """
    区域配置
    
    Attributes:
        id: 区域唯一标识
        name: 区域名称
        center: 中心位置 (x, y)
        radius: 覆盖半径 (像素)
        neighbors: 相邻区域 {方向: 区域ID}
        typical_transition_time: 典型过渡时间 (秒)
    """
    id: str
    name: str
    center: Tuple[int, int]
    radius: int
    neighbors: Dict[str, str] = field(default_factory=dict)
    typical_transition_time: float = 5.0  # 默认5秒


class ZoneLayout:
    """
    区域布局管理器
    
    管理所有照明区域的空间关系和连接信息。
    """
    
    def __init__(self):
        self.zones: Dict[str, ZoneConfig] = {}
        self._zone_positions: Dict[str, Tuple[int, int]] = {}
    
    def add_zone(self, zone: ZoneConfig):
        """添加区域"""
        self.zones[zone.id] = zone
        self._zone_positions[zone.id] = zone.center
    
    def get_zone(self, zone_id: str) -> Optional[ZoneConfig]:
        """获取区域配置"""
        return self.zones.get(zone_id)
    
    def get_all_zones(self) -> List[ZoneConfig]:
        """获取所有区域"""
        return list(self.zones.values())
    
    def get_neighbors(self, zone_id: str) -> List[str]:
        """获取相邻区域ID列表"""
        zone = self.zones.get(zone_id)
        if zone:
            return list(zone.neighbors.values())
        return []
    
    def calculate_distance(self, zone1_id: str, zone2_id: str) -> float:
        """计算两个区域中心之间的距离"""
        pos1 = self._zone_positions.get(zone1_id)
        pos2 = self._zone_positions.get(zone2_id)
        if pos1 and pos2:
            return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        return float('inf')
    
    def find_zone_by_position(self, position: Tuple[int, int]) -> Optional[str]:
        """根据位置查找所在区域"""
        for zone_id, zone in self.zones.items():
            distance = math.sqrt(
                (position[0] - zone.center[0])**2 + 
                (position[1] - zone.center[1])**2
            )
            if distance <= zone.radius:
                return zone_id
        return None
    
    def get_direction_to_zone(self, 
                              from_zone_id: str, 
                              to_zone_id: str) -> float:
        """
        计算从一个区域到另一个区域的方向角度
        
        Returns:
            方向角度 (0-360度，0为东，逆时针)
        """
        from_pos = self._zone_positions.get(from_zone_id)
        to_pos = self._zone_positions.get(to_zone_id)
        if not from_pos or not to_pos:
            return 0.0
        
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        angle = math.degrees(math.atan2(dy, dx))
        return angle % 360


class MotionPredictor:
    """
    运动预测器
    
    基于历史运动数据预测人员下一步移动方向和目标区域。
    
    Attributes:
        layout: 区域布局
        history: 运动历史记录
        confidence_threshold: 置信度阈值
    """
    
    def __init__(self, 
                 zone_layout: ZoneLayout,
                 history_size: int = 50,
                 confidence_threshold: float = 0.6):
        """
        初始化运动预测器
        
        Args:
            zone_layout: 区域布局
            history_size: 历史记录最大数量
            confidence_threshold: 置信度阈值，低于此值不预测
        """
        self.layout = zone_layout
        self.history_size = history_size
        self.confidence_threshold = confidence_threshold
        
        # 历史记录 {track_id: deque of MotionEvent}
        self.history: Dict[str, deque] = {}
        
        # 预测统计
        self.prediction_stats = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'false_predictions': 0,
        }
    
    def update(self, event: MotionEvent):
        """
        更新运动事件
        
        Args:
            event: 运动事件
        """
        if event.track_id is None:
            return
        
        if event.track_id not in self.history:
            self.history[event.track_id] = deque(maxlen=self.history_size)
        
        self.history[event.track_id].append(event)
    
    def predict_next_zone(self, 
                          current_zone: str,
                          track_id: Optional[str] = None,
                          direction: Optional[float] = None) -> Optional[str]:
        """
        预测下一个区域
        
        Args:
            current_zone: 当前区域ID
            track_id: 跟踪ID，用于获取历史轨迹
            direction: 运动方向，如不提供则从历史获取
            
        Returns:
            预测的下一个区域ID，如无法预测返回None
        """
        # 获取运动方向
        if direction is None and track_id and track_id in self.history:
            history = self.history[track_id]
            if len(history) > 0:
                direction = history[-1].direction
        
        if direction is None:
            return None
        
        # 获取相邻区域
        neighbors = self.layout.get_neighbors(current_zone)
        if not neighbors:
            return None
        
        # 计算每个相邻区域的方向
        best_match = None
        best_score = float('inf')
        
        for neighbor_id in neighbors:
            neighbor_direction = self.layout.get_direction_to_zone(
                current_zone, neighbor_id
            )
            
            # 计算方向差异 (考虑360度环绕)
            diff = abs(direction - neighbor_direction)
            diff = min(diff, 360 - diff)
            
            # 考虑历史成功率
            score = diff
            
            if score < best_score and score < 90:  # 方向差异小于90度
                best_score = score
                best_match = neighbor_id
        
        return best_match
    
    def estimate_arrival_time(self,
                             from_zone: str,
                             to_zone: str,
                             current_speed: Optional[float] = None) -> float:
        """
        估算到达目标区域的时间
        
        Args:
            from_zone: 起始区域ID
            to_zone: 目标区域ID
            current_speed: 当前速度 (m/s)，None则使用默认值
            
        Returns:
            预计到达时间 (秒)
        """
        # 获取区域配置
        from_config = self.layout.get_zone(from_zone)
        to_config = self.layout.get_zone(to_zone)
        
        if not from_config or not to_config:
            return float('inf')
        
        # 计算距离
        distance = self.layout.calculate_distance(from_zone, to_zone)
        
        # 估算速度
        if current_speed and current_speed > 0:
            speed = current_speed
        else:
            # 使用典型步行速度 1.2 m/s
            speed = 1.2
        
        # 像素到米的转换 (假设 100 像素 = 1 米)
        distance_meters = distance / 100.0
        
        # 计算时间
        if speed > 0:
            return distance_meters / speed
        
        # 使用典型过渡时间
        return from_config.typical_transition_time
    
    def predict_destination(self,
                           current_event: MotionEvent,
                           prediction_horizon: int = 3) -> List[Tuple[str, float]]:
        """
        预测目的地及概率
        
        Args:
            current_event: 当前运动事件
            prediction_horizon: 预测步数
            
        Returns:
            [(区域ID, 概率), ...] 按概率排序
        """
        if current_event.confidence < self.confidence_threshold:
            return []
        
        # 更新历史
        self.update(current_event)
        
        # 预测下一个区域
        next_zone = self.predict_next_zone(
            current_event.zone_id,
            current_event.track_id,
            current_event.direction
        )
        
        if not next_zone:
            return []
        
        # 计算到达时间
        arrival_time = self.estimate_arrival_time(
            current_event.zone_id,
            next_zone,
            current_event.speed
        )
        
        # 基于历史准确率计算概率
        confidence = current_event.confidence
        
        # 如果速度较快，概率更高
        if current_event.speed > 1.0:
            confidence *= 1.1
        
        # 限制概率范围
        confidence = min(1.0, confidence)
        
        return [(next_zone, confidence, arrival_time)]
    
    def get_motion_trend(self, 
                        track_id: str,
                        window_size: int = 5) -> Optional[Dict]:
        """
        获取运动趋势
        
        Args:
            track_id: 跟踪ID
            window_size: 时间窗口大小
            
        Returns:
            {
                'avg_direction': 平均方向,
                'avg_speed': 平均速度,
                'direction_variance': 方向方差,
                'is_consistent': 是否方向一致
            }
        """
        if track_id not in self.history:
            return None
        
        history = list(self.history[track_id])[-window_size:]
        if len(history) < 2:
            return None
        
        # 计算平均方向 (使用向量平均)
        sum_x = sum(math.cos(math.radians(e.direction)) for e in history)
        sum_y = sum(math.sin(math.radians(e.direction)) for e in history)
        avg_direction = math.degrees(math.atan2(sum_y, sum_x)) % 360
        
        # 计算平均速度
        avg_speed = sum(e.speed for e in history) / len(history)
        
        # 计算方向方差
        directions = [e.direction for e in history]
        direction_variance = self._calculate_circular_variance(directions)
        
        # 判断是否方向一致 (方差小于阈值)
        is_consistent = direction_variance < 30
        
        return {
            'avg_direction': avg_direction,
            'avg_speed': avg_speed,
            'direction_variance': direction_variance,
            'is_consistent': is_consistent,
        }
    
    def _calculate_circular_variance(self, angles: List[float]) -> float:
        """计算圆形方差 (用于角度)"""
        if not angles:
            return 0.0
        
        # 转换为向量
        vectors = [(math.cos(math.radians(a)), math.sin(math.radians(a))) 
                   for a in angles]
        
        # 计算平均向量
        avg_x = sum(v[0] for v in vectors) / len(vectors)
        avg_y = sum(v[1] for v in vectors) / len(vectors)
        
        # 计算结果向量长度 (0-1，1表示完全一致)
        r = math.sqrt(avg_x**2 + avg_y**2)
        
        # 转换为方差 (0-180，0表示完全一致)
        variance = math.degrees(math.sqrt(-2 * math.log(max(r, 0.001))))
        return variance
    
    def record_prediction_result(self, 
                                 predicted_zone: str, 
                                 actual_zone: str):
        """
        记录预测结果，用于统计准确率
        
        Args:
            predicted_zone: 预测的区域
            actual_zone: 实际的区域
        """
        self.prediction_stats['total_predictions'] += 1
        
        if predicted_zone == actual_zone:
            self.prediction_stats['correct_predictions'] += 1
        else:
            self.prediction_stats['false_predictions'] += 1
    
    def get_prediction_accuracy(self) -> float:
        """获取预测准确率"""
        total = self.prediction_stats['total_predictions']
        if total == 0:
            return 0.0
        return self.prediction_stats['correct_predictions'] / total
    
    def clear_old_history(self, max_age_seconds: float = 300):
        """清理过期历史记录"""
        now = datetime.now()
        expired_tracks = []
        
        for track_id, history in self.history.items():
            if history and (now - history[-1].timestamp).total_seconds() > max_age_seconds:
                expired_tracks.append(track_id)
        
        for track_id in expired_tracks:
            del self.history[track_id]
