#!/usr/bin/env python3
"""
区域管理模块
定义检测区域，支持多区域密度分析
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple


class Point:
    """简单点类"""
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
    
    def contains(self, x: int, y: int) -> bool:
        """检查点是否等于给定坐标"""
        return self.x == x and self.y == y


class Polygon:
    """多边形区域"""
    def __init__(self, points: List[Tuple[int, int]]):
        self.points = points
        self.contour = np.array(points, dtype=np.int32)
    
    def contains(self, x: int, y: int) -> bool:
        """检查点是否在多边形内"""
        return cv2.pointPolygonTest(self.contour, (x, y), False) >= 0
    
    def draw(self, image: np.ndarray, color: Tuple[int, int, int], 
             label: str = "", alpha: float = 0.3):
        """绘制区域"""
        overlay = image.copy()
        cv2.fillPoly(overlay, [self.contour], color)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        cv2.polylines(image, [self.contour], True, color, 2)
        
        if label:
            # 计算中心点
            moments = cv2.moments(self.contour)
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                cv2.putText(image, label, (cx - 30, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


class ZoneManager:
    """区域管理器"""
    
    def __init__(self, zones: List[Dict], frame_size: Tuple[int, int] = (640, 480)):
        """
        Args:
            zones: 区域配置列表，每项包含name和coords（归一化坐标0-1）
            frame_size: 帧尺寸 (width, height)
        """
        self.frame_size = frame_size
        self.zone_configs = zones
        self.zones: Dict[str, Dict] = {}
        self._init_zones()
    
    def _init_zones(self):
        """初始化区域"""
        w, h = self.frame_size
        
        colors = [
            (255, 100, 100),  # 红
            (100, 255, 100),  # 绿
            (100, 100, 255),  # 蓝
            (255, 255, 100),  # 黄
            (255, 100, 255),  # 紫
        ]
        
        for i, config in enumerate(self.zone_configs):
            name = config['name']
            coords = config['coords']  # 归一化坐标 [(x1,y1), (x2,y2), ...]
            
            # 转换为像素坐标
            pixel_coords = [
                (int(x * w), int(y * h)) for x, y in coords
            ]
            
            self.zones[name] = {
                'polygon': Polygon(pixel_coords),
                'color': colors[i % len(colors)],
                'name': name
            }
    
    def update_frame_size(self, frame_size: Tuple[int, int]):
        """更新帧尺寸（视频源变化时）"""
        self.frame_size = frame_size
        self._init_zones()
    
    def get_zones(self) -> Dict[str, Dict]:
        """获取所有区域"""
        return self.zones
    
    def get_zone_count(self, zone_name: str, detections: List[Dict]) -> int:
        """获取特定区域的人数"""
        if zone_name not in self.zones:
            return 0
        
        zone = self.zones[zone_name]
        count = 0
        
        for det in detections:
            if det['class'] != 'person':
                continue
            x1, y1, x2, y2 = det['bbox']
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            if zone['polygon'].contains(cx, cy):
                count += 1
        
        return count
    
    def draw_zones(self, image: np.ndarray):
        """绘制所有区域"""
        for name, zone in self.zones.items():
            zone['polygon'].draw(image, zone['color'], name, alpha=0.2)
    
    def calculate_zone_density(self, zone_name: str, detections: List[Dict]) -> float:
        """计算区域密度"""
        count = self.get_zone_count(zone_name, detections)
        
        # 简单密度估计：根据人数
        if count <= 2:
            return 0.2
        elif count <= 5:
            return 0.4
        elif count <= 10:
            return 0.6
        elif count <= 20:
            return 0.8
        else:
            return 1.0
