#!/usr/bin/env python3
"""
教室空调智能控制系统 - 主程序

功能：
- 人流密度检测与统计
- 区域热力图分析
- 空调智能启停与功率调节
- 多时段策略管理

作者：AI Assistant
"""
import sys
import time
import argparse
from pathlib import Path
from collections import deque

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np

from shared.video_capture import VideoCapture
from shared.logger import setup_logger
from classroom_ac.people_counter import PeopleCounter
from classroom_ac.zone_manager import ZoneManager
from classroom_ac.ac_controller import ACController


logger = setup_logger("classroom_ac")


class ClassroomACSystem:
    """教室空调智能控制系统"""
    
    def __init__(self, config: dict):
        self.config = config
        self.counter = PeopleCounter(
            model_path=config.get('model_path', 'models/yolov8n.onnx'),
            conf_threshold=config.get('conf_threshold', 0.5)
        )
        self.zone_manager = ZoneManager(
            zones=config.get('zones', []),
            frame_size=config.get('frame_size', (640, 480))
        )
        self.ac_controller = ACController(
            demo_mode=config.get('demo_mode', True),
            min_people=config.get('min_people', 3),
            cooldown_minutes=config.get('cooldown_minutes', 5)
        )
        self.video = None
        self.is_running = False
        
        # 历史数据用于平滑决策
        self.people_history = deque(maxlen=30)  # 30秒历史
        self.zone_history = deque(maxlen=30)
        
        # 统计
        self.stats = {
            'frames_processed': 0,
            'total_people_detected': 0,
            'ac_on_time': 0,
            'ac_off_time': 0,
            'start_time': None
        }
    
    def start(self, source):
        """启动系统"""
        logger.info("=" * 50)
        logger.info("教室空调智能控制系统启动")
        logger.info("=" * 50)
        
        self.video = VideoCapture(source)
        if not self.video.start():
            logger.error("无法启动视频源")
            return False
        
        width, height = self.video.get_size()
        self.config['frame_size'] = (width, height)
        self.zone_manager.update_frame_size((width, height))
        
        logger.info(f"视频源: {source}")
        logger.info(f"分辨率: {width}x{height}")
        
        if not self.counter.load_model():
            logger.error("模型加载失败")
            return False
        
        if not self.ac_controller.init():
            logger.error("控制器初始化失败")
            return False
        
        self.stats['start_time'] = time.time()
        self.is_running = True
        logger.info("系统初始化完成")
        return True
    
    def analyze_crowd_density(self, detections: list, frame_shape: tuple) -> dict:
        """
        分析人群密度分布
        
        Returns:
            各区域密度统计
        """
        zones = self.zone_manager.get_zones()
        zone_counts = {name: 0 for name in zones.keys()}
        
        for det in detections:
            if det['class'] != 'person':
                continue
            
            x1, y1, x2, y2 = det['bbox']
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # 判断在哪个区域
            for zone_name, zone_info in zones.items():
                if zone_info['polygon'].contains(cx, cy):
                    zone_counts[zone_name] += 1
                    break
        
        return zone_counts
    
    def draw_heatmap(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """绘制人群热力图"""
        heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
        
        for det in detections:
            if det['class'] != 'person':
                continue
            x1, y1, x2, y2 = det['bbox']
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # 在中心点画高斯分布
            cv2.circle(heatmap, (cx, cy), 50, 1.0, -1)
        
        # 模糊平滑
        heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
        
        # 归一化
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # 转色彩
        heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # 叠加
        result = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)
        
        return result
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """处理单帧"""
        # 1. 人形检测
        detections = self.counter.detect(frame)
        people_count = len([d for d in detections if d['class'] == 'person'])
        
        # 2. 区域分析
        zone_counts = self.analyze_crowd_density(detections, frame.shape)
        
        # 3. 更新历史
        self.people_history.append(people_count)
        self.zone_history.append(zone_counts)
        
        # 4. 平滑处理：取30秒平均值
        avg_people = np.mean(self.people_history) if self.people_history else 0
        
        # 5. 热力图可视化
        display_frame = self.draw_heatmap(frame, detections)
        
        # 6. 绘制检测框
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            color = (0, 255, 0) if det['class'] == 'person' else (255, 0, 0)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        
        # 7. 绘制区域
        self.zone_manager.draw_zones(display_frame)
        
        # 8. 空调控制决策
        ac_state = self.ac_controller.update(avg_people)
        
        # 9. 绘制信息面板
        self._draw_info_panel(display_frame, people_count, avg_people, zone_counts, ac_state)
        
        self.stats['frames_processed'] += 1
        self.stats['total_people_detected'] += people_count
        
        return display_frame
    
    def _draw_info_panel(self, frame: np.ndarray, current: int, avg: float, 
                         zones: dict, ac_state: bool):
        """绘制信息面板"""
        h, w = frame.shape[:2]
        
        # 背景
        panel_h = 120 + len(zones) * 25
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 文字
        y_offset = 35
        cv2.putText(frame, f"Current: {current} | Avg(30s): {avg:.1f}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        y_offset += 30
        for zone, count in zones.items():
            cv2.putText(frame, f"{zone}: {count} people", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
        
        y_offset += 10
        ac_color = (0, 255, 0) if ac_state else (0, 0, 255)
        ac_text = "AC: ON" if ac_state else "AC: OFF"
        cv2.putText(frame, ac_text, (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, ac_color, 2)
    
    def run(self):
        """主循环"""
        frame_interval = 1.0 / 10  # 10fps
        last_time = 0
        
        while self.is_running:
            frame = self.video.read()
            if frame is None:
                time.sleep(0.01)
                continue
            
            now = time.time()
            if now - last_time < frame_interval:
                continue
            last_time = now
            
            display_frame = self.process_frame(frame)
            
            cv2.imshow("Classroom AC Control", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.stop()
    
    def stop(self):
        """停止系统"""
        self.is_running = False
        
        if self.stats['start_time']:
            runtime = time.time() - self.stats['start_time']
            logger.info("=" * 50)
            logger.info("运行统计")
            logger.info(f"运行时间: {runtime/60:.1f}分钟")
            logger.info(f"处理帧数: {self.stats['frames_processed']}")
            logger.info(f"检测到人次: {self.stats['total_people_detected']}")
            logger.info("=" * 50)
        
        if self.video:
            self.video.stop()
        self.ac_controller.cleanup()
        cv2.destroyAllWindows()
        logger.info("系统已停止")


def main():
    parser = argparse.ArgumentParser(description='教室空调智能控制系统')
    parser.add_argument('--source', type=str, default='0',
                       help='视频源: 0=摄像头, 路径=视频文件')
    parser.add_argument('--mode', type=str, default='demo', choices=['demo', 'deploy'],
                       help='运行模式')
    parser.add_argument('--min-people', type=int, default=3,
                       help='开空调的最少人数阈值')
    parser.add_argument('--cooldown', type=int, default=5,
                       help='空调关闭后等待分钟数才能再开')
    
    args = parser.parse_args()
    
    config = {
        'model_path': 'models/yolov8n.onnx',
        'conf_threshold': 0.5,
        'zones': [
            {'name': 'Front', 'coords': [(0, 0), (0.5, 0), (0.5, 1), (0, 1)]},
            {'name': 'Back', 'coords': [(0.5, 0), (1, 0), (1, 1), (0.5, 1)]}
        ],
        'demo_mode': args.mode == 'demo',
        'min_people': args.min_people,
        'cooldown_minutes': args.cooldown
    }
    
    source = int(args.source) if args.source.isdigit() else args.source
    
    system = ClassroomACSystem(config)
    if system.start(source):
        try:
            system.run()
        except KeyboardInterrupt:
            logger.info("收到中断信号")
        finally:
            system.stop()
    else:
        logger.error("系统启动失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
