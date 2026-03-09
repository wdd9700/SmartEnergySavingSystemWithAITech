#!/usr/bin/env python3
"""
楼道智能灯控系统 - 主程序 v2.0
新增: 配置加载、性能监控、Web远程控制
"""
import sys
import time
import argparse
import threading
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np

from shared.video_capture import VideoCapture
from shared.logger import setup_logger
from shared.config_loader import load_config, merge_config
from shared.performance import PerformanceMonitor
from corridor_light.detector import PersonDetector
from corridor_light.enhancer import LowLightEnhancer
from corridor_light.controller import LightController


logger = setup_logger("corridor_light")


# 全局状态（供Web接口使用）
system_state = {
    'light_on': False,
    'people_count': 0,
    'brightness': 0,
    'fps': 0,
    'is_running': False
}


class WebHandler(BaseHTTPRequestHandler):
    """简单的Web控制接口"""
    
    def log_message(self, format, *args):
        pass  # 静默日志
    
    def do_GET(self):
        if self.path == '/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(system_state).encode())
        elif self.path == '/on':
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'Light ON')
        elif self.path == '/off':
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'Light OFF')
        else:
            self.send_response(404)
            self.end_headers()


def start_web_server(port=8080):
    """启动Web控制服务"""
    server = HTTPServer(('0.0.0.0', port), WebHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info(f"Web控制接口已启动: http://localhost:{port}")
    return server


class CorridorLightSystem:
    """楼道灯智能控制系统 v2"""
    
    DEFAULT_CONFIG = {
        'model_path': 'models/yolov8n.onnx',
        'conf_threshold': 0.5,
        'iou_threshold': 0.45,
        'enhance_method': 'clahe',
        'brightness_threshold': 50,
        'light_on_delay': 0,
        'light_off_delay': 5.0,
        'demo_mode': True,
        'gpio_pin': 17,
        'web_port': 8080,
        'enable_web': True,
        'show_performance': True,
        'save_video': False,
        'video_output': 'output.mp4'
    }
    
    def __init__(self, config: dict):
        self.config = {**self.DEFAULT_CONFIG, **config}
        self.detector = PersonDetector(
            model_path=self.config['model_path'],
            conf_threshold=self.config['conf_threshold'],
            iou_threshold=self.config['iou_threshold']
        )
        self.enhancer = LowLightEnhancer(
            method=self.config['enhance_method'],
            brightness_threshold=self.config['brightness_threshold']
        )
        self.controller = LightController(
            light_on_delay=self.config['light_on_delay'],
            light_off_delay=self.config['light_off_delay'],
            demo_mode=self.config['demo_mode'],
            gpio_pin=self.config['gpio_pin']
        )
        self.video = None
        self.is_running = False
        self.monitor = PerformanceMonitor()
        self.video_writer = None
        
        # Web服务
        self.web_server = None
        if self.config.get('enable_web'):
            self.web_server = start_web_server(self.config.get('web_port', 8080))
        
        # 统计
        self.stats = {
            'frames_processed': 0,
            'people_detected_total': 0,
            'light_on_count': 0,
            'light_on_time': 0,
            'last_light_on': None,
            'start_time': None
        }
    
    def start(self, source) -> bool:
        """启动系统"""
        logger.info("=" * 60)
        logger.info("楼道智能灯控系统 v2.0 启动")
        logger.info("=" * 60)
        logger.info(f"模式: {'Demo' if self.config['demo_mode'] else 'Deploy'}")
        logger.info(f"增强算法: {self.config['enhance_method']}")
        logger.info(f"关灯延迟: {self.config['light_off_delay']}s")
        
        # 初始化视频源
        self.video = VideoCapture(source)
        if not self.video.start():
            logger.error("无法启动视频源")
            return False
        
        width, height = self.video.get_size()
        logger.info(f"视频源: {source} | 分辨率: {width}x{height}")
        
        # 初始化视频写入
        if self.config.get('save_video'):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.config['video_output'], fourcc, 15, (width, height)
            )
        
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
        system_state['is_running'] = True
        
        logger.info("系统初始化完成，开始运行")
        logger.info("按 'q' 退出 | 's' 保存截图 | 'f' 强制开关灯")
        return True
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """处理单帧"""
        start_time = time.time()
        original = frame.copy()
        
        # 1. 检测光照
        brightness = self.enhancer.estimate_brightness(frame)
        need_enhance = brightness < self.config['brightness_threshold']
        
        # 2. 视频增强
        if need_enhance:
            enhanced = self.enhancer.enhance(frame)
            display_frame = enhanced.copy()
            detection_frame = enhanced
        else:
            display_frame = frame.copy()
            detection_frame = frame
        
        # 3. 人形检测
        inference_start = time.time()
        detections = self.detector.detect(detection_frame)
        inference_time = (time.time() - inference_start) * 1000
        
        person_count = len([d for d in detections if d['class'] == 'person'])
        
        # 4. 绘制检测框
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            label = f"{det['class']} {conf:.2f}"
            color = (0, 255, 0) if det['class'] == 'person' else (255, 0, 0)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_frame, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 5. 灯光控制
        light_state = self.controller.update(person_count > 0)
        
        # 更新统计
        if light_state and not self.stats['last_light_on']:
            self.stats['light_on_count'] += 1
            self.stats['last_light_on'] = time.time()
        elif not light_state and self.stats['last_light_on']:
            self.stats['light_on_time'] += time.time() - self.stats['last_light_on']
            self.stats['last_light_on'] = None
        
        # 6. 绘制信息面板
        self._draw_info_panel(display_frame, brightness, person_count, light_state, inference_time)
        
        # 7. 性能监控
        self.monitor.record_frame(inference_time)
        if self.config.get('show_performance'):
            perf_lines = self.monitor.draw_overlay(display_frame)
            for i, line in enumerate(perf_lines):
                cv2.putText(display_frame, line, (display_frame.shape[1] - 200, 30 + i*25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 更新全局状态
        system_state['light_on'] = light_state
        system_state['people_count'] = person_count
        system_state['brightness'] = int(brightness)
        system_state['fps'] = round(self.monitor.stats.fps, 1)
        
        self.stats['frames_processed'] += 1
        self.stats['people_detected_total'] += person_count
        
        return display_frame
    
    def _draw_info_panel(self, frame, brightness, person_count, light_state, inference_time):
        """绘制信息面板"""
        h, w = frame.shape[:2]
        
        # 半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 状态信息
        y = 35
        cv2.putText(frame, f"Brightness: {brightness:.0f} {'LOW' if brightness < 50 else 'OK'}", 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        y += 30
        cv2.putText(frame, f"People: {person_count} | Inference: {inference_time:.1f}ms",
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        y += 30
        light_color = (0, 255, 0) if light_state else (0, 0, 255)
        light_text = "LIGHT: ON" if light_state else "LIGHT: OFF"
        cv2.putText(frame, light_text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, light_color, 2)
        
        y += 30
        mode_text = "DEMO MODE" if self.config['demo_mode'] else "DEPLOY MODE"
        cv2.putText(frame, mode_text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
    
    def run(self):
        """主循环"""
        frame_interval = 1.0 / 15  # 15fps
        last_time = 0
        screenshot_count = 0
        
        while self.is_running:
            frame = self.video.read()
            if frame is None:
                time.sleep(0.01)
                continue
            
            now = time.time()
            if now - last_time < frame_interval:
                continue
            last_time = now
            
            # 处理帧
            display_frame = self.process_frame(frame)
            
            # 保存视频
            if self.video_writer:
                self.video_writer.write(display_frame)
            
            # 显示
            cv2.imshow("Corridor Light Control", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_count += 1
                filename = f"screenshot_{screenshot_count:03d}.jpg"
                cv2.imwrite(filename, display_frame)
                logger.info(f"截图保存: {filename}")
            elif key == ord('f'):
                # 强制开关灯
                self.controller.force_on() if not system_state['light_on'] else self.controller.force_off()
        
        self.stop()
    
    def stop(self):
        """停止系统"""
        self.is_running = False
        system_state['is_running'] = False
        
        # 打印统计
        if self.stats['start_time']:
            runtime = time.time() - self.stats['start_time']
            perf_summary = self.monitor.get_summary()
            
            logger.info("=" * 60)
            logger.info("运行统计")
            logger.info(f"运行时间: {runtime/60:.1f}分钟")
            logger.info(f"处理帧数: {self.stats['frames_processed']}")
            logger.info(f"平均FPS: {perf_summary['avg_fps']:.1f}")
            logger.info(f"平均推理时间: {perf_summary['avg_inference_ms']:.1f}ms")
            logger.info(f"开灯次数: {self.stats['light_on_count']}")
            logger.info(f"开灯总时长: {self.stats['light_on_time']:.1f}秒")
            logger.info("=" * 60)
        
        if self.video:
            self.video.stop()
        if self.video_writer:
            self.video_writer.release()
        self.controller.cleanup()
        cv2.destroyAllWindows()
        logger.info("系统已停止")


def main():
    parser = argparse.ArgumentParser(description='楼道智能灯控系统 v2.0')
    parser.add_argument('--source', type=str, default=None,
                       help='视频源: 0=摄像头, 路径=视频文件')
    parser.add_argument('--mode', type=str, default=None, choices=['demo', 'deploy'],
                       help='运行模式: demo=演示(无硬件), deploy=部署(控制实际灯光)')
    parser.add_argument('--conf', type=float, default=None,
                       help='检测置信度阈值')
    parser.add_argument('--off-delay', type=float, default=None,
                       help='人离开后关灯延迟(秒)')
    parser.add_argument('--config', type=str, default='corridor_light/config.yaml',
                       help='配置文件路径')
    parser.add_argument('--no-web', action='store_true',
                       help='禁用Web控制接口')
    parser.add_argument('--save-video', action='store_true',
                       help='保存处理后的视频')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config, CorridorLightSystem.DEFAULT_CONFIG)
    config = merge_config(config, args)
    
    if args.no_web:
        config['enable_web'] = False
    
    # 解析视频源
    source = config.get('source', 0)
    if args.source is not None:
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
