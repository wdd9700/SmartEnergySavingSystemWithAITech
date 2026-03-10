#!/usr/bin/env python3
"""
楼道智能灯控系统 - 主程序 v3.0
新增: 基于人形位置的智能灯光区间控制
- 根据脚底位置确定人所在灯区域
- 只开启人所在和人前方的灯
- 人离开区域后立即关闭对应灯
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
from corridor_light.light_zones import LightConfig, create_default_config
from corridor_light.zone_controller import ZoneLightController


logger = setup_logger("corridor_light")


# 全局状态（供Web接口使用）
system_state = {
    'active_lights': [],
    'people_count': 0,
    'people_locations': [],
    'brightness': 0,
    'fps': 0,
    'is_running': False,
    'mode': 'zone_based'  # zone_based | calibration
}


class WebHandler(BaseHTTPRequestHandler):
    """Web控制接口"""
    
    def log_message(self, format, *args):
        pass
    
    def do_GET(self):
        if self.path == '/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(system_state).encode())
        elif self.path == '/lights':
            # 获取当前开启的灯
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'active_lights': system_state['active_lights']}).encode())
        else:
            self.send_response(404)
            self.end_headers()


def start_web_server(port=8080):
    """启动Web控制服务"""
    server = HTTPServer(('0.0.0.0', port), WebHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info(f"Web控制接口: http://localhost:{port}")
    return server


class CorridorLightSystemV3:
    """
    楼道灯智能控制系统 v3
    基于人形位置的灯光区间控制
    """
    
    DEFAULT_CONFIG = {
        'model_path': 'models/yolov8n.onnx',
        'conf_threshold': 0.5,
        'iou_threshold': 0.45,
        'enhance_method': 'clahe',
        'brightness_threshold': 50,
        'light_off_delay': 0.5,  # 人离开后延迟关灯
        'facing_direction': 'forward',  # forward/backward/both
        'demo_mode': True,
        'web_port': 8080,
        'enable_web': True,
        'show_performance': True,
        'save_video': False,
        'video_output': 'output.mp4',
        'light_config_path': 'corridor_light/light_config.json',
        'calibration_mode': False,  # 校准模式
        'zone_radius': 100,  # 自动校准时的区域半径
    }
    
    def __init__(self, config: dict):
        self.config = {**self.DEFAULT_CONFIG, **config}
        
        # 视频源
        self.video = None
        self.video_writer = None
        self.is_running = False
        self.monitor = PerformanceMonitor()
        
        # 检测器和增强器
        self.detector = PersonDetector(
            model_path=self.config['model_path'],
            conf_threshold=self.config['conf_threshold'],
            iou_threshold=self.config['iou_threshold']
        )
        self.enhancer = LowLightEnhancer(
            method=self.config['enhance_method'],
            brightness_threshold=self.config['brightness_threshold']
        )
        
        # 灯光配置和控制器
        self.light_config = None
        self.controller = None
        self._load_light_config()
        
        # Web服务
        self.web_server = None
        if self.config.get('enable_web'):
            self.web_server = start_web_server(self.config.get('web_port', 8080))
        
        # 统计
        self.stats = {
            'frames_processed': 0,
            'people_detected_total': 0,
            'start_time': None
        }
        
        # 可视化
        self.show_zones = True
        self.show_foot_points = True
    
    def _load_light_config(self):
        """加载灯光配置"""
        config_path = self.config.get('light_config_path')
        
        if config_path and Path(config_path).exists():
            try:
                self.light_config = LightConfig.load_from_file(config_path)
                logger.info(f"已加载灯光配置: {config_path}")
            except Exception as e:
                logger.warning(f"加载配置失败: {e}，使用默认配置")
                self.light_config = create_default_config()
        else:
            logger.info("使用默认灯光配置")
            self.light_config = create_default_config()
        
        # 创建控制器
        self.controller = ZoneLightController(
            light_config=self.light_config,
            light_off_delay=self.config['light_off_delay'],
            facing_direction=self.config['facing_direction'],
            demo_mode=self.config['demo_mode']
        )
    
    def start(self, source) -> bool:
        """启动系统"""
        logger.info("=" * 60)
        logger.info("楼道智能灯控系统 v3.0 - 基于位置的灯光控制")
        logger.info("=" * 60)
        logger.info(f"模式: {'Demo' if self.config['demo_mode'] else 'Deploy'}")
        logger.info(f"增强算法: {self.config['enhance_method']}")
        logger.info(f"关灯延迟: {self.config['light_off_delay']}s")
        logger.info(f"默认朝向: {self.config['facing_direction']}")
        
        if self.config.get('calibration_mode'):
            logger.info("⚠️  校准模式: 按 'c' 校准灯光位置")
        
        # 初始化视频源
        self.video = VideoCapture(source)
        if not self.video.start():
            logger.error("无法启动视频源")
            return False
        
        width, height = self.video.get_size()
        logger.info(f"视频源: {source} | 分辨率: {width}x{height}")
        
        # 根据实际分辨率更新默认配置
        if not self.config.get('light_config_path') or not Path(self.config['light_config_path']).exists():
            self.light_config = create_default_config(width, height)
            self.controller.config = self.light_config
        
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
        
        logger.info("\n系统初始化完成!")
        logger.info("快捷键: q=退出 | c=校准灯光 | s=截图 | z=切换区域显示 | f=切换脚底点显示")
        return True
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """处理单帧"""
        start_time = time.time()
        h, w = frame.shape[:2]
        
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
        
        # 只保留人形
        person_detections = [d for d in detections if d['class'] == 'person']
        
        # 4. 灯光控制
        light_states = self.controller.update(person_detections)
        active_lights = [lid for lid, state in light_states.items() if state]
        
        # 5. 绘制检测结果
        display_frame = self._draw_detections(display_frame, person_detections, active_lights)
        
        # 6. 绘制灯光区域
        if self.show_zones:
            display_frame = self._draw_light_zones(display_frame)
        
        # 7. 绘制信息面板
        display_frame = self._draw_info_panel(
            display_frame, brightness, len(person_detections), 
            active_lights, inference_time
        )
        
        # 8. 性能监控
        self.monitor.record_frame(inference_time)
        if self.config.get('show_performance'):
            perf_lines = self.monitor.draw_overlay(display_frame)
            for i, line in enumerate(perf_lines):
                cv2.putText(display_frame, line, (w - 200, 30 + i*25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 更新全局状态
        system_state['active_lights'] = active_lights
        system_state['people_count'] = len(person_detections)
        system_state['people_locations'] = [
            {'id': i, 'x': d['foot_point'][0], 'y': d['foot_point'][1]}
            for i, d in enumerate(person_detections)
        ]
        system_state['brightness'] = int(brightness)
        system_state['fps'] = round(self.monitor.stats.fps, 1)
        
        self.stats['frames_processed'] += 1
        self.stats['people_detected_total'] += len(person_detections)
        
        return display_frame
    
    def _draw_detections(self, frame, detections, active_lights):
        """绘制检测结果"""
        # 获取活跃的灯中心点（用于连线）
        active_centers = {}
        for light_id in active_lights:
            zone = self.controller.config.get_zone(light_id)
            if zone:
                active_centers[light_id] = (zone.x, zone.y)
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            foot_x, foot_y = det['foot_point']
            conf = det['confidence']
            
            # 检测框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 标签
            label = f"Person {conf:.2f}"
            cv2.putText(frame, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 脚底点
            if self.show_foot_points:
                cv2.circle(frame, (foot_x, foot_y), 5, (0, 0, 255), -1)
                cv2.circle(frame, (foot_x, foot_y), 8, (0, 0, 255), 2)
                
                # 连线到激活的灯
                for light_id, center in active_centers.items():
                    zone = self.controller.config.get_zone(light_id)
                    if zone and zone.contains_point((foot_x, foot_y)):
                        # 人在该区域内，画实线
                        cv2.line(frame, (foot_x, foot_y), center, (0, 255, 255), 2)
                    else:
                        # 前方灯，画虚线（用短实线模拟）
                        cv2.line(frame, (foot_x, foot_y), center, (128, 128, 128), 1)
        
        return frame
    
    def _draw_light_zones(self, frame):
        """绘制灯光区域"""
        for zone in self.controller.config.get_all_zones():
            # 判断是否开启
            is_active = self.controller.light_states.get(zone.id, {}).get('state', False)
            
            # 颜色: 开启=绿色, 关闭=灰色
            color = (0, 255, 0) if is_active else (128, 128, 128)
            thickness = 2 if is_active else 1
            
            # 绘制覆盖区域
            cv2.circle(frame, (zone.x, zone.y), zone.radius, color, thickness)
            
            # 绘制灯位置
            cv2.circle(frame, (zone.x, zone.y), 8, color, -1)
            
            # 绘制名称
            cv2.putText(frame, zone.name, (zone.x - 30, zone.y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # 绘制前方关联（箭头）
            for forward_id in zone.forward_zones:
                forward_zone = self.controller.config.get_zone(forward_id)
                if forward_zone:
                    # 画箭头指向
                    self._draw_arrow(frame, (zone.x, zone.y), 
                                    (forward_zone.x, forward_zone.y), 
                                    (64, 64, 64), 1)
        
        return frame
    
    def _draw_arrow(self, frame, start, end, color, thickness):
        """绘制箭头"""
        cv2.line(frame, start, end, color, thickness)
        # 简化的箭头
        angle = np.arctan2(end[1] - start[1], end[0] - start[0])
        arrow_len = 10
        arrow_angle = np.pi / 6
        
        x1 = int(end[0] - arrow_len * np.cos(angle - arrow_angle))
        y1 = int(end[1] - arrow_len * np.sin(angle - arrow_angle))
        x2 = int(end[0] - arrow_len * np.cos(angle + arrow_angle))
        y2 = int(end[1] - arrow_len * np.sin(angle + arrow_angle))
        
        cv2.line(frame, end, (x1, y1), color, thickness)
        cv2.line(frame, end, (x2, y2), color, thickness)
    
    def _draw_info_panel(self, frame, brightness, people_count, active_lights, inference_time):
        """绘制信息面板"""
        h, w = frame.shape[:2]
        
        # 半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (420, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 状态信息
        y = 35
        cv2.putText(frame, f"Brightness: {brightness:.0f} {'LOW' if brightness < 50 else 'OK'}", 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        y += 30
        cv2.putText(frame, f"People: {people_count} | Inference: {inference_time:.1f}ms",
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        y += 30
        lights_text = f"Active Lights: {len(active_lights)}"
        if active_lights:
            zone_names = [self.controller.config.get_zone(lid).name 
                         for lid in active_lights 
                         if self.controller.config.get_zone(lid)]
            lights_text += f" ({', '.join(zone_names)})"
        light_color = (0, 255, 0) if active_lights else (0, 0, 255)
        cv2.putText(frame, lights_text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, light_color, 2)
        
        y += 30
        mode_text = "ZONE MODE" if not self.config.get('calibration_mode') else "CALIBRATION MODE"
        cv2.putText(frame, mode_text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        
        return frame
    
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
            cv2.imshow("Corridor Light Control v3", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_count += 1
                filename = f"screenshot_{screenshot_count:03d}.jpg"
                cv2.imwrite(filename, display_frame)
                logger.info(f"截图保存: {filename}")
            elif key == ord('c'):
                # 校准模式
                self._calibrate_lights()
            elif key == ord('z'):
                self.show_zones = not self.show_zones
                logger.info(f"区域显示: {'开启' if self.show_zones else '关闭'}")
            elif key == ord('f'):
                self.show_foot_points = not self.show_foot_points
                logger.info(f"脚底点显示: {'开启' if self.show_foot_points else '关闭'}")
        
        self.stop()
    
    def _calibrate_lights(self):
        """从当前帧校准灯光位置"""
        logger.info("开始灯光校准...")
        
        # 获取当前帧
        frame = self.video.read()
        if frame is None:
            logger.warning("无法获取帧")
            return
        
        # 检测人形
        detections = self.detector.detect(frame)
        person_detections = [d for d in detections if d['class'] == 'person']
        
        if not person_detections:
            logger.warning("未检测到人形，无法进行校准")
            return
        
        # 执行校准
        save_path = self.config.get('light_config_path', 'corridor_light/light_config.json')
        success = self.controller.calibrate_from_frame(
            person_detections,
            zone_radius=self.config.get('zone_radius', 100),
            save_path=save_path
        )
        
        if success:
            logger.info(f"校准完成! 配置已保存到 {save_path}")
    
    def stop(self):
        """停止系统"""
        self.is_running = False
        system_state['is_running'] = False
        
        # 打印统计
        if self.stats['start_time']:
            runtime = time.time() - self.stats['start_time']
            perf_summary = self.monitor.get_summary()
            controller_stats = self.controller.get_stats()
            
            logger.info("=" * 60)
            logger.info("运行统计")
            logger.info(f"运行时间: {runtime/60:.1f}分钟")
            logger.info(f"处理帧数: {self.stats['frames_processed']}")
            logger.info(f"平均FPS: {perf_summary['avg_fps']:.1f}")
            logger.info(f"平均推理时间: {perf_summary['avg_inference_ms']:.1f}ms")
            logger.info(f"\n区域进入次数:")
            for zone_id, count in controller_stats['zone_entries'].items():
                zone = self.controller.config.get_zone(zone_id)
                name = zone.name if zone else zone_id
                logger.info(f"  {name}: {count}次")
            logger.info(f"\n开灯统计:")
            for zone_id, count in controller_stats['light_on_count'].items():
                zone = self.controller.config.get_zone(zone_id)
                name = zone.name if zone else zone_id
                on_time = controller_stats['light_on_time'].get(zone_id, 0)
                logger.info(f"  {name}: {count}次, 总时长{on_time:.1f}s")
            logger.info("=" * 60)
        
        if self.video:
            self.video.stop()
        if self.video_writer:
            self.video_writer.release()
        self.controller.cleanup()
        cv2.destroyAllWindows()
        logger.info("系统已停止")


def main():
    parser = argparse.ArgumentParser(description='楼道智能灯控系统 v3.0 - 基于位置的灯光控制')
    parser.add_argument('--source', type=str, default=None,
                       help='视频源: 0=摄像头, 路径=视频文件')
    parser.add_argument('--mode', type=str, default=None, choices=['demo', 'deploy'],
                       help='运行模式: demo=演示, deploy=部署')
    parser.add_argument('--conf', type=float, default=None,
                       help='检测置信度阈值')
    parser.add_argument('--off-delay', type=float, default=None,
                       help='人离开后关灯延迟(秒)')
    parser.add_argument('--facing', type=str, default=None, 
                       choices=['forward', 'backward', 'both'],
                       help='默认朝向: forward=前方灯, backward=后方灯, both=双向')
    parser.add_argument('--config', type=str, default='corridor_light/config.yaml',
                       help='配置文件路径')
    parser.add_argument('--light-config', type=str, default=None,
                       help='灯光配置文件路径')
    parser.add_argument('--calibrate', action='store_true',
                       help='启动时进入校准模式')
    parser.add_argument('--no-web', action='store_true',
                       help='禁用Web控制接口')
    parser.add_argument('--save-video', action='store_true',
                       help='保存处理后的视频')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config, CorridorLightSystemV3.DEFAULT_CONFIG)
    config = merge_config(config, args)
    
    if args.no_web:
        config['enable_web'] = False
    if args.calibrate:
        config['calibration_mode'] = True
    if args.light_config:
        config['light_config_path'] = args.light_config
    
    # 解析视频源
    source = config.get('source', 0)
    if args.source is not None:
        source = int(args.source) if args.source.isdigit() else args.source
    
    # 启动系统
    system = CorridorLightSystemV3(config)
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
