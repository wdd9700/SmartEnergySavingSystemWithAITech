#!/usr/bin/env python3
"""
楼道智能灯控系统 - 主程序

功能：
- 夜间低光照视频增强
- 人形检测（YOLOv8）
- 智能灯光控制（人来灯亮，人走灯灭）
- 支持Demo模式（无硬件）和实际部署模式

作者：AI Assistant
"""
import sys
import time
import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np

from shared.video_capture import VideoCapture
from shared.logger import setup_logger
from corridor_light.detector import PersonDetector
from corridor_light.enhancer import LowLightEnhancer
from corridor_light.controller import LightController


logger = setup_logger("corridor_light")


class CorridorLightSystem:
    """楼道灯智能控制系统"""
    
    def __init__(self, config: dict):
        self.config = config
        self.detector = PersonDetector(
            model_path=config.get('model_path', 'models/yolov8n.onnx'),
            conf_threshold=config.get('conf_threshold', 0.5),
            iou_threshold=config.get('iou_threshold', 0.45)
        )
        self.enhancer = LowLightEnhancer(
            method=config.get('enhance_method', 'clahe'),
            brightness_threshold=config.get('brightness_threshold', 50)
        )
        self.controller = LightController(
            light_on_delay=config.get('light_on_delay', 0),
            light_off_delay=config.get('light_off_delay', 5.0),
            demo_mode=config.get('demo_mode', True)
        )
        self.video = None
        self.is_running = False
        
        # 统计
        self.stats = {
            'frames_processed': 0,
            'people_detected_total': 0,
            'light_on_count': 0,
            'start_time': None
        }
    
    def start(self, source):
        """启动系统"""
        logger.info("=" * 50)
        logger.info("楼道智能灯控系统启动")
        logger.info("=" * 50)
        
        # 初始化视频源
        self.video = VideoCapture(source)
        if not self.video.start():
            logger.error("无法启动视频源")
            return False
        
        width, height = self.video.get_size()
        logger.info(f"视频源: {source}")
        logger.info(f"分辨率: {width}x{height}")
        
        # 初始化检测器
        if not self.detector.load_model():
            logger.error("模型加载失败")
            return False
        
        # 初始化控制器
        if not self.controller.init():
            logger.error("控制器初始化失败")
            return False
        
        self.stats['start_time'] = time.time()
        self.is_running = True
        logger.info("系统初始化完成，开始运行")
        return True
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """处理单帧"""
        original = frame.copy()
        
        # 1. 检测光照条件
        brightness = self.enhancer.estimate_brightness(frame)
        need_enhance = brightness < self.config.get('brightness_threshold', 50)
        
        # 2. 视频增强（仅在低光照时）
        if need_enhance:
            enhanced = self.enhancer.enhance(frame)
            display_frame = enhanced.copy()
            detection_frame = enhanced
        else:
            display_frame = frame.copy()
            detection_frame = frame
        
        # 3. 人形检测
        detections = self.detector.detect(detection_frame)
        person_count = len([d for d in detections if d['class'] == 'person'])
        
        # 4. 绘制检测结果
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            label = f"{det['class']} {conf:.2f}"
            color = (0, 255, 0) if det['class'] == 'person' else (255, 0, 0)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_frame, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 5. 灯光控制逻辑
        light_state = self.controller.update(person_count > 0)
        
        # 6. 绘制状态信息
        status_text = f"Brightness: {brightness:.0f} | People: {person_count} | Light: {'ON' if light_state else 'OFF'}"
        cv2.putText(display_frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 增强提示
        if need_enhance:
            cv2.putText(display_frame, "LOW LIGHT - ENHANCED", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        # 更新统计
        self.stats['frames_processed'] += 1
        self.stats['people_detected_total'] += person_count
        
        return display_frame
    
    def run(self):
        """主循环"""
        frame_processor = type('obj', (object,), {'frame_interval': 1.0/15, 'last_time': 0})()
        
        while self.is_running:
            frame = self.video.read()
            if frame is None:
                time.sleep(0.01)
                continue
            
            # 限流：15fps检测足够
            now = time.time()
            if now - frame_processor.last_time < frame_processor.frame_interval:
                continue
            frame_processor.last_time = now
            
            # 处理帧
            display_frame = self.process_frame(frame)
            
            # 显示
            cv2.imshow("Corridor Light Control", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.stop()
    
    def stop(self):
        """停止系统"""
        self.is_running = False
        
        # 打印统计
        if self.stats['start_time']:
            runtime = time.time() - self.stats['start_time']
            logger.info("=" * 50)
            logger.info("运行统计")
            logger.info(f"运行时间: {runtime:.1f}s")
            logger.info(f"处理帧数: {self.stats['frames_processed']}")
            logger.info(f"检测到人: {self.stats['people_detected_total']} 次")
            logger.info("=" * 50)
        
        if self.video:
            self.video.stop()
        self.controller.cleanup()
        cv2.destroyAllWindows()
        logger.info("系统已停止")


def main():
    parser = argparse.ArgumentParser(description='楼道智能灯控系统')
    parser.add_argument('--source', type=str, default='0',
                       help='视频源: 0=摄像头, 路径=视频文件')
    parser.add_argument('--mode', type=str, default='demo', choices=['demo', 'deploy'],
                       help='运行模式: demo=演示(无硬件), deploy=部署(控制实际灯光)')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='检测置信度阈值')
    parser.add_argument('--off-delay', type=float, default=5.0,
                       help='人离开后关灯延迟(秒)')
    
    args = parser.parse_args()
    
    # 配置
    config = {
        'model_path': 'models/yolov8n.onnx',
        'conf_threshold': args.conf,
        'iou_threshold': 0.45,
        'enhance_method': 'clahe',
        'brightness_threshold': 50,
        'light_on_delay': 0,
        'light_off_delay': args.off_delay,
        'demo_mode': args.mode == 'demo'
    }
    
    # 解析视频源
    source = int(args.source) if args.source.isdigit() else args.source
    
    # 启动系统
    system = CorridorLightSystem(config)
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
