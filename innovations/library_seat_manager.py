#!/usr/bin/env python3
"""
创新用途1: 图书馆座位智能管理系统

功能:
- 实时统计各楼层/区域座位占用情况
- 学生可查看空座位分布
- 占座检测（物品无人超过30分钟）
- 热力图引导学生到空闲区域

技术栈复用:
- YOLOv8检测人形和物品
- 区域管理模块
- 视频增强模块
"""
import sys
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np

from shared.video_capture import VideoCapture
from shared.logger import setup_logger
from shared.performance import PerformanceMonitor
from corridor_light.detector import PersonDetector
from classroom_ac.zone_manager import ZoneManager


logger = setup_logger("library_seat")


class Seat:
    """单个座位状态"""
    def __init__(self, seat_id, zone, bbox):
        self.id = seat_id
        self.zone = zone
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.status = 'empty'  # empty/occupied/suspicious
        self.occupied_since = None
        self.vacant_since = datetime.now()
        self.occupant_id = None
        
    def update(self, has_person, has_item):
        """更新座位状态"""
        now = datetime.now()
        
        if has_person:
            if self.status != 'occupied':
                self.status = 'occupied'
                self.occupied_since = now
                self.vacant_since = None
        elif has_item:
            # 有物品但无人 - 可疑占座
            if self.status == 'occupied':
                # 人刚离开
                self.vacant_since = now
                self.status = 'suspicious'
            elif self.status == 'suspicious':
                # 检查是否超过30分钟
                if self.vacant_since and (now - self.vacant_since).seconds > 1800:
                    self.status = 'empty'  # 释放座位
                    self.vacant_since = now
        else:
            # 完全空
            self.status = 'empty'
            self.occupied_since = None
            self.vacant_since = now
    
    def get_info(self):
        """获取座位信息"""
        return {
            'id': self.id,
            'zone': self.zone,
            'status': self.status,
            'occupied_duration': (datetime.now() - self.occupied_since).seconds if self.occupied_since else 0,
            'vacant_duration': (datetime.now() - self.vacant_since).seconds if self.vacant_since else 0
        }


class LibrarySeatManager:
    """图书馆座位管理系统"""
    
    def __init__(self, config):
        self.config = config
        self.detector = PersonDetector(
            model_path=config.get('model_path', 'models/yolov8n.onnx'),
            conf_threshold=0.4
        )
        self.video = None
        self.is_running = False
        self.monitor = PerformanceMonitor()
        
        # 座位定义 (可通过配置文件或初始化时手动标注)
        self.seats = []
        self._init_seats()
        
        # 统计
        self.daily_stats = {
            'total_checkins': 0,
            'peak_occupancy': 0,
            'avg_session_minutes': 0,
            'sessions': []
        }
        
        # API数据缓存
        self.api_data = {
            'timestamp': None,
            'total_seats': 0,
            'occupied': 0,
            'empty': 0,
            'suspicious': 0,
            'zones': {}
        }
    
    def _init_seats(self):
        """初始化座位布局"""
        # 示例：定义一个阅览室的座位布局
        # 实际部署时可通过配置文件加载或通过Web界面标注
        seat_layout = self.config.get('seat_layout', [])
        
        if not seat_layout:
            # 默认示例布局
            seat_layout = [
                # 区域A
                {'id': 'A1', 'zone': 'Zone-A', 'bbox': [100, 100, 200, 200]},
                {'id': 'A2', 'zone': 'Zone-A', 'bbox': [220, 100, 320, 200]},
                {'id': 'A3', 'zone': 'Zone-A', 'bbox': [340, 100, 440, 200]},
                # 区域B
                {'id': 'B1', 'zone': 'Zone-B', 'bbox': [100, 250, 200, 350]},
                {'id': 'B2', 'zone': 'Zone-B', 'bbox': [220, 250, 320, 350]},
            ]
        
        for seat_def in seat_layout:
            self.seats.append(Seat(
                seat_def['id'], 
                seat_def['zone'], 
                seat_def['bbox']
            ))
        
        logger.info(f"初始化完成: {len(self.seats)} 个座位")
    
    def detect_items(self, frame, seat_bbox):
        """检测座位上是否有物品（简化版）"""
        x1, y1, x2, y2 = seat_bbox
        seat_roi = frame[y1:y2, x1:x2]
        
        if seat_roi.size == 0:
            return False
        
        # 方法：通过颜色变化和边缘检测判断是否有物品
        gray = cv2.cvtColor(seat_roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # 如果边缘密度超过阈值，认为有物品
        return edge_density > 0.05
    
    def update_seats(self, frame, detections):
        """更新所有座位状态"""
        for seat in self.seats:
            # 检查是否有人
            has_person = False
            person_center = None
            
            for det in detections:
                if det['class'] != 'person':
                    continue
                px1, py1, px2, py2 = det['bbox']
                pcx, pcy = (px1 + px2) // 2, (py1 + py2) // 2
                
                # 检查人是否在座位区域内
                sx1, sy1, sx2, sy2 = seat.bbox
                if sx1 < pcx < sx2 and sy1 < pcy < sy2:
                    has_person = True
                    person_center = (pcx, pcy)
                    break
            
            # 检查是否有物品
            has_item = self.detect_items(frame, seat.bbox) if not has_person else False
            
            # 更新座位状态
            old_status = seat.status
            seat.update(has_person, has_item)
            
            # 统计
            if old_status != 'occupied' and seat.status == 'occupied':
                self.daily_stats['total_checkins'] += 1
    
    def draw_seats(self, frame):
        """绘制座位状态"""
        for seat in self.seats:
            x1, y1, x2, y2 = seat.bbox
            
            # 根据状态选择颜色
            if seat.status == 'occupied':
                color = (0, 0, 255)  # 红色 - 占用
                label = f"{seat.id}: {seat.zone}"
            elif seat.status == 'suspicious':
                color = (0, 165, 255)  # 橙色 - 可疑占座
                vacant_mins = (datetime.now() - seat.vacant_since).seconds // 60 if seat.vacant_since else 0
                label = f"{seat.id}: ? {vacant_mins}min"
            else:
                color = (0, 255, 0)  # 绿色 - 空闲
                label = f"{seat.id}: FREE"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def update_api_data(self):
        """更新API数据缓存"""
        zones = defaultdict(lambda: {'total': 0, 'occupied': 0, 'empty': 0})
        
        for seat in self.seats:
            zones[seat.zone]['total'] += 1
            if seat.status == 'occupied':
                zones[seat.zone]['occupied'] += 1
            elif seat.status == 'empty':
                zones[seat.zone]['empty'] += 1
        
        self.api_data = {
            'timestamp': datetime.now().isoformat(),
            'total_seats': len(self.seats),
            'occupied': sum(1 for s in self.seats if s.status == 'occupied'),
            'empty': sum(1 for s in self.seats if s.status == 'empty'),
            'suspicious': sum(1 for s in self.seats if s.status == 'suspicious'),
            'zones': dict(zones)
        }
    
    def get_recommendation(self) -> str:
        """获取座位推荐"""
        empty_seats = [s for s in self.seats if s.status == 'empty']
        
        if not empty_seats:
            return "当前暂无空座位"
        
        # 找到空座位最多的区域
        zone_counts = defaultdict(int)
        for seat in empty_seats:
            zone_counts[seat.zone] += 1
        
        best_zone = max(zone_counts.items(), key=lambda x: x[1])
        
        return f"推荐前往 {best_zone[0]} 区域，有 {best_zone[1]} 个空座位"
    
    def start(self, source):
        """启动系统"""
        logger.info("图书馆座位管理系统启动")
        
        self.video = VideoCapture(source)
        if not self.video.start():
            logger.error("无法启动视频源")
            return False
        
        if not self.detector.load_model():
            logger.error("模型加载失败")
            return False
        
        self.is_running = True
        logger.info(f"管理 {len(self.seats)} 个座位")
        return True
    
    def run(self):
        """主循环"""
        frame_count = 0
        
        while self.is_running:
            frame = self.video.read()
            if frame is None:
                continue
            
            frame_count += 1
            if frame_count % 5 != 0:  # 每5帧处理一次
                continue
            
            # 检测
            detections = self.detector.detect(frame)
            
            # 更新座位状态
            self.update_seats(frame, detections)
            
            # 更新API数据
            if frame_count % 30 == 0:  # 每秒更新一次API数据
                self.update_api_data()
            
            # 可视化
            display = frame.copy()
            self.draw_seats(display)
            
            # 绘制统计信息
            info = self.api_data
            y = 30
            cv2.putText(display, f"Total: {info['total_seats']} | Occupied: {info['occupied']} | Empty: {info['empty']}",
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            y += 30
            recommendation = self.get_recommendation()
            cv2.putText(display, recommendation, (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imshow("Library Seat Manager", display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.stop()
    
    def stop(self):
        """停止系统"""
        self.is_running = False
        if self.video:
            self.video.stop()
        cv2.destroyAllWindows()
        
        # 打印日报
        logger.info("=" * 50)
        logger.info("今日统计")
        logger.info(f"总入馆人次: {self.daily_stats['total_checkins']}")
        logger.info(f"峰值同时在馆: {self.daily_stats['peak_occupancy']}")
        logger.info("=" * 50)


def main():
    """独立运行入口"""
    config = {
        'model_path': 'models/yolov8n.onnx',
        'source': 0
    }
    
    manager = LibrarySeatManager(config)
    if manager.start(config['source']):
        manager.run()


if __name__ == "__main__":
    main()
