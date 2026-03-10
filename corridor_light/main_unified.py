#!/usr/bin/env python3
"""
楼道智能灯控系统 - 统一主程序 (集成 v1/v2/v3/v4 功能)
支持:
- 传统模式: 单灯整体控制
- 区域模式: 基于人形位置的区间控制 (v3)
- 自动校准: 基于亮度分析的自动位置校准 (v4)
- 数据记录: 检测日志、热力图、能耗统计
"""
import sys
import time
import argparse
import threading
from pathlib import Path
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np

from shared.video_capture import VideoCapture
from shared.logger import setup_logger
from shared.data_recorder import DataRecorder, HeatmapGenerator, EnergyEstimator
from corridor_light.detector import PersonDetector
from corridor_light.enhancer import LowLightEnhancer
from corridor_light.controller import LightController
from corridor_light.light_zones import LightConfig, create_default_config
from corridor_light.zone_controller import ZoneLightController


logger = setup_logger("corridor_light")


# 全局状态
system_state = {
    'mode': 'traditional',  # traditional | zone_based | calibration
    'is_running': False,
    'light_on': False,
    'active_lights': [],
    'people_count': 0,
    'brightness': 0,
    'fps': 0,
    'energy_stats': {},
    'statistics': {}
}


class WebHandler(BaseHTTPRequestHandler):
    """Web API接口"""
    
    def log_message(self, format, *args):
        pass
    
    def do_GET(self):
        if self.path == '/status':
            self.send_json(system_state)
        elif self.path == '/stats':
            self.send_json(system_state.get('statistics', {}))
        elif self.path == '/energy':
            self.send_json(system_state.get('energy_stats', {}))
        elif self.path.startswith('/export/'):
            # 导出数据
            hours = int(self.path.split('/')[-1]) if '/' in self.path else 24
            recorder = getattr(self.server, 'recorder', None)
            if recorder:
                path = recorder.export_to_json(f'export_{hours}h.json', hours)
                self.send_json({'exported_to': path})
            else:
                self.send_json({'error': 'Recorder not available'}, 503)
        else:
            self.send_json({'error': 'Not found'}, 404)
    
    def send_json(self, data, code=200):
        self.send_response(code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode())


def start_web_server(port=8080, recorder=None):
    """启动Web服务"""
    server = HTTPServer(('0.0.0.0', port), WebHandler)
    server.recorder = recorder
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info(f"Web API: http://localhost:{port}")
    return server


class CorridorLightSystem:
    """
    统一楼道灯控系统
    支持多种工作模式
    """
    
    DEFAULT_CONFIG = {
        'model_path': 'models/yolov8n.onnx',
        'conf_threshold': 0.5,
        'iou_threshold': 0.45,
        'enhance_method': 'clahe',
        'brightness_threshold': 50,
        
        # 传统模式参数
        'light_on_delay': 0,
        'light_off_delay': 5.0,
        
        # 区域模式参数
        'zone_off_delay': 0.5,
        'facing_direction': 'forward',
        
        # 通用参数
        'demo_mode': True,
        'gpio_pin': 17,
        'web_port': 8080,
        'enable_web': True,
        'show_performance': True,
        'show_zones': True,
        'save_video': False,
        'video_output': 'output.mp4',
        
        # 数据记录参数
        'enable_recording': True,
        'log_dir': 'logs',
        'generate_heatmap': True,
        'heatmap_interval': 300,  # 每5分钟保存热力图
        
        # 配置文件
        'light_config_path': 'corridor_light/light_config.json',
        'zone_radius': 100,
    }
    
    def __init__(self, config: dict, mode: str = 'traditional'):
        self.config = {**self.DEFAULT_CONFIG, **config}
        self.mode = mode
        
        # 视频
        self.video = None
        self.video_writer = None
        self.is_running = False
        
        # 检测和增强
        self.detector = PersonDetector(
            model_path=self.config['model_path'],
            conf_threshold=self.config['conf_threshold'],
            iou_threshold=self.config['iou_threshold']
        )
        self.enhancer = LowLightEnhancer(
            method=self.config['enhance_method'],
            brightness_threshold=self.config['brightness_threshold']
        )
        
        # 灯光控制 (根据模式选择)
        self.traditional_controller = None
        self.zone_controller = None
        self._init_controllers()
        
        # 数据记录
        self.recorder = None
        self.heatmap = None
        self.energy = None
        if self.config.get('enable_recording'):
            self.recorder = DataRecorder(log_dir=self.config['log_dir'])
            self.energy = EnergyEstimator()
        
        # Web服务
        self.web_server = None
        if self.config.get('enable_web'):
            self.web_server = start_web_server(
                self.config.get('web_port', 8080),
                self.recorder
            )
        
        # 统计
        self.stats = {
            'frames_processed': 0,
            'people_detected_total': 0,
            'start_time': None
        }
        
        # 显示选项
        self.show_zones = self.config.get('show_zones', True)
        self.show_foot_points = True
        
        # 热力图
        self.last_heatmap_time = 0
    
    def _init_controllers(self):
        """初始化灯光控制器"""
        # 传统控制器
        self.traditional_controller = LightController(
            light_on_delay=self.config['light_on_delay'],
            light_off_delay=self.config['light_off_delay'],
            demo_mode=self.config['demo_mode'],
            gpio_pin=self.config['gpio_pin']
        )
        
        # 区域控制器
        light_config = self._load_light_config()
        self.zone_controller = ZoneLightController(
            light_config=light_config,
            light_off_delay=self.config['zone_off_delay'],
            facing_direction=self.config['facing_direction'],
            demo_mode=self.config['demo_mode']
        )
    
    def _load_light_config(self) -> LightConfig:
        """加载灯光配置"""
        config_path = self.config.get('light_config_path')
        
        if config_path and Path(config_path).exists():
            try:
                return LightConfig.load_from_file(config_path)
            except Exception as e:
                logger.warning(f"加载配置失败: {e}")
        
        return create_default_config()
    
    def start(self, source) -> bool:
        """启动系统"""
        logger.info("=" * 60)
        logger.info(f"楼道智能灯控系统 - {self.mode} 模式")
        logger.info("=" * 60)
        
        # 启动视频
        self.video = VideoCapture(source)
        if not self.video.start():
            logger.error("无法启动视频源")
            return False
        
        width, height = self.video.get_size()
        logger.info(f"视频源: {source} | 分辨率: {width}x{height}")
        
        # 初始化热力图
        if self.config.get('generate_heatmap'):
            self.heatmap = HeatmapGenerator((height, width))
        
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
        if not self.traditional_controller.init():
            logger.error("传统控制器初始化失败")
            return False
        
        if not self.zone_controller.init():
            logger.error("区域控制器初始化失败")
            return False
        
        # 更新区域控制器配置（根据实际分辨率）
        if not Path(self.config.get('light_config_path', '')).exists():
            default_config = create_default_config(width, height)
            self.zone_controller.config = default_config
        
        self.stats['start_time'] = time.time()
        self.is_running = True
        system_state['is_running'] = True
        system_state['mode'] = self.mode
        
        logger.info(f"\n系统启动成功!")
        logger.info(f"工作模式: {self.mode}")
        logger.info(f"数据记录: {'开启' if self.recorder else '关闭'}")
        logger.info(f"\n快捷键: q=退出 | m=切换模式 | s=截图 | z=切换区域显示")
        
        return True
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """处理单帧"""
        start_time = time.time()
        h, w = frame.shape[:2]
        timestamp = datetime.now()
        
        # 1. 检测光照
        brightness = self.enhancer.estimate_brightness(frame)
        need_enhance = brightness < self.config['brightness_threshold']
        
        # 2. 视频增强
        if need_enhance:
            enhanced = self.enhancer.enhance(frame)
            detection_frame = enhanced
        else:
            detection_frame = frame
        
        display_frame = frame.copy()
        
        # 3. 人形检测
        inference_start = time.time()
        detections = self.detector.detect(detection_frame)
        inference_time = (time.time() - inference_start) * 1000
        
        person_detections = [d for d in detections if d['class'] == 'person']
        person_count = len(person_detections)
        
        # 4. 灯光控制 (根据模式)
        light_states = {}
        active_lights = []
        
        if self.mode == 'traditional':
            # 传统模式: 单灯整体控制
            light_on = self.traditional_controller.update(person_count > 0)
            light_states = {'main_light': light_on}
            active_lights = ['main_light'] if light_on else []
            
            # 能耗统计
            if self.energy:
                self.energy.update_light_state('main_light', light_on)
        
        elif self.mode == 'zone_based':
            # 区域模式: 基于位置的区间控制
            light_states = self.zone_controller.update(person_detections)
            active_lights = [lid for lid, state in light_states.items() if state]
            
            # 能耗统计
            if self.energy:
                for lid, state in light_states.items():
                    self.energy.update_light_state(lid, state)
        
        # 5. 绘制检测结果
        self._draw_detections(display_frame, person_detections, active_lights)
        
        # 6. 绘制灯光区域 (区域模式)
        if self.mode == 'zone_based' and self.show_zones:
            self._draw_light_zones(display_frame)
        
        # 7. 绘制信息面板
        fps = 1000.0 / (inference_time + 1e-6)
        self._draw_info_panel(display_frame, brightness, person_count, 
                             active_lights, inference_time, fps)
        
        # 8. 数据记录
        if self.recorder:
            person_locations = [d['foot_point'] for d in person_detections if 'foot_point' in d]
            
            self.recorder.record_detection(
                timestamp=timestamp,
                camera_id='main',
                people_count=person_count,
                light_states=light_states,
                brightness=brightness,
                inference_time_ms=inference_time,
                fps=fps,
                person_locations=person_locations
            )
            
            # 热力图
            if self.heatmap and person_locations:
                self.heatmap.add_frame(person_locations)
            
            # 定期保存热力图
            if self.heatmap and (time.time() - self.last_heatmap_time) > self.config.get('heatmap_interval', 300):
                heatmap_path = f"{self.config['log_dir']}/heatmap_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
                self.heatmap.save(heatmap_path, background=frame)
                self.last_heatmap_time = time.time()
                logger.info(f"热力图已保存: {heatmap_path}")
        
        # 9. 更新全局状态
        system_state['light_on'] = bool(active_lights)
        system_state['active_lights'] = active_lights
        system_state['people_count'] = person_count
        system_state['brightness'] = int(brightness)
        system_state['fps'] = round(fps, 1)
        
        if self.energy:
            system_state['energy_stats'] = self.energy.get_statistics()
        
        if self.recorder and self.stats['frames_processed'] % 300 == 0:
            system_state['statistics'] = self.recorder.get_statistics(hours=1)
        
        self.stats['frames_processed'] += 1
        self.stats['people_detected_total'] += person_count
        
        return display_frame
    
    def _draw_detections(self, frame, detections, active_lights):
        """绘制检测结果"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            
            # 检测框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 标签
            label = f"Person {conf:.2f}"
            cv2.putText(frame, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 脚底点
            if self.show_foot_points and 'foot_point' in det:
                foot_x, foot_y = det['foot_point']
                cv2.circle(frame, (foot_x, foot_y), 5, (0, 0, 255), -1)
                cv2.circle(frame, (foot_x, foot_y), 8, (0, 0, 255), 2)
    
    def _draw_light_zones(self, frame):
        """绘制灯光区域"""
        for zone in self.zone_controller.config.get_all_zones():
            is_active = self.zone_controller.light_states.get(zone.id, {}).get('state', False)
            color = (0, 255, 0) if is_active else (128, 128, 128)
            thickness = 2 if is_active else 1
            
            cv2.circle(frame, (zone.x, zone.y), zone.radius, color, thickness)
            cv2.circle(frame, (zone.x, zone.y), 8, color, -1)
            cv2.putText(frame, zone.name, (zone.x - 30, zone.y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def _draw_info_panel(self, frame, brightness, people_count, 
                        active_lights, inference_time, fps):
        """绘制信息面板"""
        h, w = frame.shape[:2]
        
        # 半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (420, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        y = 35
        cv2.putText(frame, f"Mode: {self.mode.upper()}", 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        y += 30
        cv2.putText(frame, f"Brightness: {brightness:.0f} {'LOW' if brightness < 50 else 'OK'}", 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        y += 30
        cv2.putText(frame, f"People: {people_count} | FPS: {fps:.1f}",
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        y += 30
        lights_text = f"Lights: {len(active_lights)}"
        if active_lights:
            lights_text += f" ({', '.join(active_lights[:3])})"
        light_color = (0, 255, 0) if active_lights else (0, 0, 255)
        cv2.putText(frame, lights_text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, light_color, 2)
        
        y += 30
        if self.energy:
            energy_stats = self.energy.get_statistics()
            energy_text = f"Energy: {energy_stats.get('total_energy_wh', 0):.2f} Wh"
            cv2.putText(frame, energy_text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
    
    def switch_mode(self):
        """切换工作模式"""
        if self.mode == 'traditional':
            self.mode = 'zone_based'
        else:
            self.mode = 'traditional'
        
        system_state['mode'] = self.mode
        logger.info(f"切换到模式: {self.mode}")
    
    def run(self):
        """主循环"""
        frame_interval = 1.0 / 15
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
                logger.info(f"截图: {filename}")
            elif key == ord('m'):
                self.switch_mode()
            elif key == ord('z'):
                self.show_zones = not self.show_zones
        
        self.stop()
    
    def stop(self):
        """停止系统"""
        self.is_running = False
        system_state['is_running'] = False
        
        # 最终统计
        if self.stats['start_time']:
            runtime = time.time() - self.stats['start_time']
            logger.info("=" * 60)
            logger.info("运行统计")
            logger.info(f"运行时间: {runtime/60:.1f}分钟")
            logger.info(f"处理帧数: {self.stats['frames_processed']}")
            
            if self.energy:
                stats = self.energy.get_statistics()
                logger.info(f"总能耗: {stats['total_energy_wh']:.2f} Wh")
            
            if self.recorder:
                stats = self.recorder.get_statistics(hours=24)
                logger.info(f"平均人数: {stats.get('people', {}).get('avg', 0):.2f}")
                logger.info(f"最大人数: {stats.get('people', {}).get('max', 0)}")
            
            logger.info("=" * 60)
        
        # 保存最终热力图
        if self.heatmap:
            final_heatmap = f"{self.config['log_dir']}/heatmap_final.jpg"
            self.heatmap.save(final_heatmap)
            logger.info(f"最终热力图: {final_heatmap}")
        
        if self.recorder:
            self.recorder.close()
        
        if self.video:
            self.video.stop()
        if self.video_writer:
            self.video_writer.release()
        
        self.traditional_controller.cleanup()
        self.zone_controller.cleanup()
        cv2.destroyAllWindows()
        
        logger.info("系统已停止")


def main():
    parser = argparse.ArgumentParser(description='楼道智能灯控系统 - 统一主程序')
    parser.add_argument('--source', type=str, default='0',
                       help='视频源 (0=摄像头, 路径=视频文件)')
    parser.add_argument('--mode', type=str, default='traditional',
                       choices=['traditional', 'zone_based'],
                       help='工作模式: traditional=传统单灯控制, zone_based=区域区间控制')
    parser.add_argument('--demo', action='store_true',
                       help='演示模式 (无实际硬件控制)')
    parser.add_argument('--off-delay', type=float, default=None,
                       help='关灯延迟(秒)')
    parser.add_argument('--config', type=str, default='corridor_light/config.yaml',
                       help='配置文件路径')
    parser.add_argument('--no-web', action='store_true',
                       help='禁用Web接口')
    parser.add_argument('--no-record', action='store_true',
                       help='禁用数据记录')
    
    args = parser.parse_args()
    
    # 解析视频源
    source = int(args.source) if args.source.isdigit() else args.source
    
    # 配置
    config = {'demo_mode': args.demo}
    if args.off_delay is not None:
        config['light_off_delay'] = args.off_delay
    if args.no_web:
        config['enable_web'] = False
    if args.no_record:
        config['enable_recording'] = False
    
    # 启动
    system = CorridorLightSystem(config, mode=args.mode)
    
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
