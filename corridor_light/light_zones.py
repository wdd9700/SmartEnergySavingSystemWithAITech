#!/usr/bin/env python3
"""
灯光区域配置模块
定义每个灯的位置、覆盖区域和关联关系
支持基于人形位置的智能灯光控制
"""
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass
class LightZone:
    """灯光区域定义"""
    id: str                    # 灯的唯一标识
    name: str                  # 灯名称
    x: int                     # 灯在画面中的x位置(像素)
    y: int                     # 灯在画面中的y位置(像素)
    radius: int                # 覆盖半径(像素)
    forward_zones: List[str]   # 前方区域(人面向方向)的灯ID列表
    backward_zones: List[str]  # 后方区域的灯ID列表
    
    def contains_point(self, point: Tuple[int, int]) -> bool:
        """判断点是否在该灯覆盖区域内"""
        px, py = point
        dx = px - self.x
        dy = py - self.y
        return (dx * dx + dy * dy) <= self.radius * self.radius
    
    def distance_to_point(self, point: Tuple[int, int]) -> float:
        """计算点到灯中心的距离"""
        px, py = point
        return ((px - self.x) ** 2 + (py - self.y) ** 2) ** 0.5


class LightConfig:
    """灯光配置管理器"""
    
    def __init__(self, zones: List[LightZone] = None):
        self.zones: Dict[str, LightZone] = {}
        if zones:
            for zone in zones:
                self.zones[zone.id] = zone
    
    def add_zone(self, zone: LightZone):
        """添加灯光区域"""
        self.zones[zone.id] = zone
    
    def get_zone(self, zone_id: str) -> Optional[LightZone]:
        """获取指定区域"""
        return self.zones.get(zone_id)
    
    def get_all_zones(self) -> List[LightZone]:
        """获取所有区域"""
        return list(self.zones.values())
    
    def find_zone_by_position(self, point: Tuple[int, int]) -> Optional[LightZone]:
        """根据位置查找所在区域"""
        for zone in self.zones.values():
            if zone.contains_point(point):
                return zone
        return None
    
    def find_nearest_zone(self, point: Tuple[int, int]) -> Optional[LightZone]:
        """查找最近的区域"""
        if not self.zones:
            return None
        return min(self.zones.values(), key=lambda z: z.distance_to_point(point))
    
    def get_lights_for_person(self, foot_point: Tuple[int, int], 
                              facing_direction: str = 'forward') -> List[str]:
        """
        根据人形位置获取应开启的灯
        
        Args:
            foot_point: 脚底位置 (x, y)
            facing_direction: 'forward'=前进方向, 'backward'=后退方向, 'both'=双向
        
        Returns:
            应开启的灯ID列表
        """
        lights_to_turn_on = set()
        
        # 1. 找到人所在的区域
        current_zone = self.find_zone_by_position(foot_point)
        
        if current_zone:
            # 人所在区域的灯必须开启
            lights_to_turn_on.add(current_zone.id)
            
            # 2. 根据朝向开启前方/后方的灯
            if facing_direction in ['forward', 'both']:
                for zone_id in current_zone.forward_zones:
                    lights_to_turn_on.add(zone_id)
            
            if facing_direction in ['backward', 'both']:
                for zone_id in current_zone.backward_zones:
                    lights_to_turn_on.add(zone_id)
        else:
            # 人不在任何定义区域内，开启最近的灯及其前方灯
            nearest = self.find_nearest_zone(foot_point)
            if nearest:
                lights_to_turn_on.add(nearest.id)
                for zone_id in nearest.forward_zones:
                    lights_to_turn_on.add(zone_id)
        
        return list(lights_to_turn_on)
    
    def calibrate_from_detections(self, detections: List[Dict], 
                                   zone_radius: int = 100) -> List[LightZone]:
        """
        从检测人形位置自动校准灯光配置
        
        Args:
            detections: 检测结果列表，每项包含 'foot_point'
            zone_radius: 自动生成的区域半径
        
        Returns:
            生成的灯光区域列表
        """
        zones = []
        used_points = []
        
        for i, det in enumerate(detections):
            foot_x, foot_y = det['foot_point']
            
            # 检查是否太接近已有点
            too_close = False
            for ux, uy in used_points:
                if ((foot_x - ux) ** 2 + (foot_y - uy) ** 2) < (zone_radius * 0.5) ** 2:
                    too_close = True
                    break
            
            if not too_close:
                zone_id = f"light_{len(zones)}"
                zone = LightZone(
                    id=zone_id,
                    name=f"灯{len(zones)+1}",
                    x=foot_x,
                    y=foot_y,
                    radius=zone_radius,
                    forward_zones=[],  # 稍后手动设置
                    backward_zones=[]
                )
                zones.append(zone)
                used_points.append((foot_x, foot_y))
        
        # 自动推断前后关系 (按x坐标排序，假定走廊是水平的)
        if len(zones) > 1:
            sorted_zones = sorted(zones, key=lambda z: z.x)
            for i, zone in enumerate(sorted_zones):
                # 前方的灯 (x坐标更大的)
                if i < len(sorted_zones) - 1:
                    zone.forward_zones = [sorted_zones[i+1].id]
                # 后方的灯
                if i > 0:
                    zone.backward_zones = [sorted_zones[i-1].id]
        
        # 添加到配置
        for zone in zones:
            self.add_zone(zone)
        
        return zones
    
    def save_to_file(self, filepath: str):
        """保存配置到JSON文件"""
        data = []
        for zone in self.zones.values():
            data.append({
                'id': zone.id,
                'name': zone.name,
                'x': zone.x,
                'y': zone.y,
                'radius': zone.radius,
                'forward_zones': zone.forward_zones,
                'backward_zones': zone.backward_zones
            })
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'LightConfig':
        """从JSON文件加载配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        zones = []
        for item in data:
            zones.append(LightZone(**item))
        
        return cls(zones)


# 预定义的示例配置
def create_default_config(frame_width: int = 640, frame_height: int = 480) -> LightConfig:
    """
    创建默认灯光配置 (适用于标准走廊场景)
    
    假设走廊从左到右排列3盏灯
    """
    config = LightConfig()
    
    # 灯1 - 左侧
    config.add_zone(LightZone(
        id="light_0",
        name="入口灯",
        x=int(frame_width * 0.2),
        y=int(frame_height * 0.5),
        radius=80,
        forward_zones=["light_1"],
        backward_zones=[]
    ))
    
    # 灯2 - 中间
    config.add_zone(LightZone(
        id="light_1",
        name="中间灯",
        x=int(frame_width * 0.5),
        y=int(frame_height * 0.5),
        radius=80,
        forward_zones=["light_2"],
        backward_zones=["light_0"]
    ))
    
    # 灯3 - 右侧
    config.add_zone(LightZone(
        id="light_2",
        name="出口灯",
        x=int(frame_width * 0.8),
        y=int(frame_height * 0.5),
        radius=80,
        forward_zones=[],
        backward_zones=["light_1"]
    ))
    
    return config
