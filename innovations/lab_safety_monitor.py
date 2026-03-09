#!/usr/bin/env python3
"""
创新用途3: 实验室安全监控系统

功能:
- 实验人员安全防护检测（未穿实验服、未戴护目镜）
- 危险区域越界检测
- 紧急情况检测（人员倒地、烟雾）
- 自动告警和通知

适用场景:
- 化学实验室
- 物理实验室  
- 生物实验室
"""
import sys
import time
from pathlib import Path
from datetime import datetime
from enum import Enum
from collections import deque

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np

from shared.video_capture import VideoCapture
from shared.logger import setup_logger
from corridor_light.detector import PersonDetector


logger = setup_logger("lab_safety")


class SafetyLevel(Enum):
    """安全等级"""
    SAFE = 0
    WARNING = 1
    DANGER = 2
    EMERGENCY = 3


class SafetyAlert:
    """安全告警"""
    def __init__(self, level: SafetyLevel, message: str, zone: str = ""):
        self.timestamp = datetime.now()
        self.level = level
        self.message = message
        self.zone = zone
        self.acknowledged = False


class LabSafetyMonitor:
    """实验室安全监控"""
    
    # 安全装备检测颜色范围 (HSV格式)
    SAFETY_COLORS = {
        'lab_coat': {
            'lower': np.array([0, 0, 200]),    # 白色
            'upper': np.array([180, 30, 255])
        },
        'goggles': {
            'lower': np.array([100, 50, 50]),  # 蓝色护目镜
            'upper': np.array([130, 255, 255])
        }
    }
    
    def __init__(self, config):
        self.config = config
        self.detector = PersonDetector(
            model_path=config.get('model_path', 'models/yolov8n.onnx'),
            conf_threshold=0.5
        )
        self.video = None
        self.is_running = False
        
        # 危险区域
        self.danger_zones = config.get('danger_zones', [])
        
        # 告警管理
        self.alerts = deque(maxlen=100)
        self.active_alerts = []
        
        # 统计
        self.violation_count = defaultdict(int)
        self.last_alert_time = {}
        
        # 告警冷却时间（避免重复告警）
        self.alert_cooldown = 30  # 秒
    
    def detect_safety_equipment(self, frame, person_bbox) -> Dict:
        """检测安全装备"""
        x1, y1, x2, y2 = person_bbox
        person_roi = frame[y1:y2, x1:x2]
        
        if person_roi.size == 0:
            return {'lab_coat': False, 'goggles': False}
        
        results = {}
        hsv = cv2.cvtColor(person_roi, cv2.COLOR_BGR2HSV)
        
        # 检测实验服 (白色区域占比)
        white_mask = cv2.inRange(hsv, 
                                  self.SAFETY_COLORS['lab_coat']['lower'],
                                  self.SAFETY_COLORS['lab_coat']['upper'])
        white_ratio = np.sum(white_mask > 0) / white_mask.size
        results['lab_coat'] = white_ratio > 0.3  # 30%以上为白色
        
        # 检测护目镜 (头部区域蓝色)
        head_y = y1 + (y2 - y1) // 4
        head_roi = frame[y1:head_y, x1:x2]
        if head_roi.size > 0:
            head_hsv = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)
            blue_mask = cv2.inRange(head_hsv,
                                     self.SAFETY_COLORS['goggles']['lower'],
                                     self.SAFETY_COLORS['goggles']['upper'])
            blue_ratio = np.sum(blue_mask > 0) / blue_mask.size
            results['goggles'] = blue_ratio > 0.05
        else:
            results['goggles'] = False
        
        return results
    
    def detect_fall(self, person_bbox, previous_bboxes) -> bool:
        """检测人员倒地"""
        if len(previous_bboxes) < 5:
            return False
        
        x1, y1, x2, y2 = person_bbox
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / max(height, 1)
        
        # 宽高比大于2认为可能倒地
        if aspect_ratio > 2.0:
            # 检查是否持续倒地
            fall_frames = 0
            for prev_bbox in list(previous_bboxes)[-5:]:
                px1, py1, px2, py2 = prev_bbox
                p_ratio = (px2 - px1) / max(py2 - py1, 1)
                if p_ratio > 1.5:
                    fall_frames += 1
            
            return fall_frames >= 3
        
        return False
    
    def detect_smoke(self, frame) -> float:
        """检测烟雾 (简化版)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # 烟雾通常表现为高亮度低纹理区域
        _, bright = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
        bright_ratio = np.sum(bright > 0) / bright.size
        
        # 边缘检测，烟雾区域边缘少
        edges = cv2.Canny(blur, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # 高亮度 + 低边缘 = 可能是烟雾
        smoke_score = bright_ratio * (1 - edge_density * 10)
        
        return smoke_score
    
    def check_danger_zone(self, person_bbox) -> Optional[str]:
        """检查是否进入危险区域"""
        x1, y1, x2, y2 = person_bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        for zone in self.danger_zones:
            zx1, zy1, zx2, zy2 = zone['bbox']
            if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                return zone['name']
        
        return None
    
    def trigger_alert(self, level: SafetyLevel, message: str, zone: str = ""):
        """触发告警"""
        alert_key = f"{level.name}_{message}_{zone}"
        now = time.time()
        
        # 检查冷却时间
        if alert_key in self.last_alert_time:
            if now - self.last_alert_time[alert_key] < self.alert_cooldown:
                return
        
        self.last_alert_time[alert_key] = now
        
        alert = SafetyAlert(level, message, zone)
        self.alerts.append(alert)
        self.active_alerts.append(alert)
        
        # 记录违规
        self.violation_count[message] += 1
        
        # 日志记录
        if level == SafetyLevel.EMERGENCY:
            logger.error(f"🚨 紧急告警: {message} {zone}")
        elif level == SafetyLevel.DANGER:
            logger.warning(f"⚠️ 危险: {message} {zone}")
        elif level == SafetyLevel.WARNING:
            logger.info(f"⚡ 警告: {message} {zone}")
    
    def process_frame(self, frame, person_history):
        """处理单帧"""
        display = frame.copy()
        
        # 1. 人员检测
        detections = self.detector.detect(frame)
        people = [d for d in detections if d['class'] == 'person']
        
        # 2. 烟雾检测
        smoke_score = self.detect_smoke(frame)
        if smoke_score > 0.3:
            self.trigger_alert(SafetyLevel.EMERGENCY, "检测到烟雾", "实验室")
            cv2.putText(display, "⚠️ SMOKE DETECTED", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        # 3. 处理每个人
        for i, person in enumerate(people):
            bbox = person['bbox']
            x1, y1, x2, y2 = bbox
            
            # 更新历史
            person_id = i  # 简化处理，实际应使用跟踪器
            if person_id not in person_history:
                person_history[person_id] = deque(maxlen=10)
            person_history[person_id].append(bbox)
            
            # 安全装备检测
            safety = self.detect_safety_equipment(frame, bbox)
            
            # 倒地检测
            if self.detect_fall(bbox, person_history[person_id]):
                self.trigger_alert(SafetyLevel.EMERGENCY, "人员倒地", f"人员{person_id}")
                color = (0, 0, 255)
            # 危险区域
            elif danger_zone := self.check_danger_zone(bbox):
                self.trigger_alert(SafetyLevel.DANGER, f"进入危险区域", danger_zone)
                color = (0, 165, 255)
            # 装备不全
            elif not safety['lab_coat'] or not safety['goggles']:
                missing = []
                if not safety['lab_coat']:
                    missing.append('实验服')
                if not safety['goggles']:
                    missing.append('护目镜')
                self.trigger_alert(SafetyLevel.WARNING, f"未穿戴: {', '.join(missing)}")
                color = (0, 255, 255)
            else:
                color = (0, 255, 0)
            
            # 绘制
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            
            # 状态标签
            status_text = []
            if not safety['lab_coat']:
                status_text.append("无实验服")
            if not safety['goggles']:
                status_text.append("无护目镜")
            
            if status_text:
                label = " | ".join(status_text)
                cv2.putText(display, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 4. 绘制危险区域
        for zone in self.danger_zones:
            x1, y1, x2, y2 = zone['bbox']
            overlay = display.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(display, zone['name'], (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 5. 绘制告警列表
        y_offset = 100
        for alert in list(self.active_alerts)[-3:]:
            time_str = alert.timestamp.strftime('%H:%M:%S')
            if alert.level == SafetyLevel.EMERGENCY:
                color = (0, 0, 255)
                prefix = "🚨"
            elif alert.level == SafetyLevel.DANGER:
                color = (0, 165, 255)
                prefix = "⚠️"
            else:
                color = (0, 255, 255)
                prefix = "⚡"
            
            text = f"{prefix} {time_str} {alert.message}"
            cv2.putText(display, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            y_offset += 25
        
        # 清理已确认的旧告警
        self.active_alerts = [a for a in self.active_alerts 
                             if not a.acknowledged and 
                             (datetime.now() - a.timestamp).seconds < 300]
        
        return display
    
    def start(self, source):
        """启动监控"""
        logger.info("实验室安全监控系统启动")
        
        self.video = VideoCapture(source)
        if not self.video.start():
            logger.error("无法启动视频源")
            return False
        
        if not self.detector.load_model():
            logger.error("模型加载失败")
            return False
        
        self.is_running = True
        logger.info(f"监控 {len(self.danger_zones)} 个危险区域")
        return True
    
    def run(self):
        """主循环"""
        person_history = {}
        
        while self.is_running:
            frame = self.video.read()
            if frame is None:
                continue
            
            display = self.process_frame(frame, person_history)
            
            cv2.imshow("Lab Safety Monitor", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                # 确认所有告警
                for alert in self.active_alerts:
                    alert.acknowledged = True
                self.active_alerts = []
        
        self.stop()
    
    def stop(self):
        """停止监控"""
        self.is_running = False
        if self.video:
            self.video.stop()
        cv2.destroyAllWindows()
        
        # 生成安全报告
        logger.info("=" * 50)
        logger.info("安全监控报告")
        logger.info(f"总告警数: {len(self.alerts)}")
        logger.info("违规统计:")
        for violation, count in self.violation_count.items():
            logger.info(f"  - {violation}: {count}次")
        logger.info("=" * 50)


def main():
    """独立运行入口"""
    config = {
        'model_path': 'models/yolov8n.onnx',
        'danger_zones': [
            {'name': 'Chemical Zone', 'bbox': [400, 200, 600, 400]},
            {'name': 'High Voltage', 'bbox': [100, 300, 250, 450]}
        ]
    }
    
    monitor = LabSafetyMonitor(config)
    if monitor.start(0):
        monitor.run()


if __name__ == "__main__":
    main()
