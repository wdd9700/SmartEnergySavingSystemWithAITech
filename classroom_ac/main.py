#!/usr/bin/env python3
"""
教室空调智能控制系统 - 主程序 v2.0
新增: 配置加载、性能监控、时段策略、节能报告
"""
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np

from shared.video_capture import VideoCapture
from shared.logger import setup_logger
from shared.config_loader import load_config, merge_config
from shared.performance import PerformanceMonitor
from classroom_ac.people_counter import PeopleCounter
from classroom_ac.zone_manager import ZoneManager
from classroom_ac.ac_controller import ACController, ACMode


logger = setup_logger("classroom_ac")


class ClassroomACSystem:
    """教室空调智能控制系统 v2"""
    
    DEFAULT_CONFIG = {
        'model_path': 'models/yolov8n.onnx',
        'conf_threshold': 0.5,
        'iou_threshold': 0.45,
        'zones': [
            {'name': 'Front', 'coords': [[0, 0], [0.5, 0], [0.5, 1], [0, 1]]},
            {'name': 'Back', 'coords': [[0.5, 0], [1, 0], [1, 1], [0.5, 1]]}
        ],
        'demo_mode': True,
        'min_people': 3,
        'cooldown_minutes': 5,
        'target_temp_default': 26,
        'gpio_pin': 18,
        'show_performance': True,
        'save_video': False,
        'video_output': 'classroom_output.mp4',
        # 时段策略
        'time_policy': {
            'enable': True,
            'class_hours': [
                {'start': '08:00', 'end': '12:00'},
                {'start': '14:00', 'end': '18:00'},
                {'start': '19:00', 'end': '22:00'}
            ],
            'temp_adjust': {
                'morning': 26,
                'afternoon': 25,
                'evening': 26
            }
        }
    }
    
    def __init__(self, config: dict):
        self.config = {**self.DEFAULT_CONFIG, **config}
        self.counter = PeopleCounter(
            model_path=self.config['model_path'],
            conf_threshold=self.config['conf_threshold']
        )
        self.zone_manager = ZoneManager(
            zones=self.config['zones'],
            frame_size=(640, 480)
        )
        self.ac_controller = ACController(
            demo_mode=self.config['demo_mode'],
            min_people=self.config['min_people'],
            cooldown_minutes=self.config['cooldown_minutes']
        )
        self.video = None
        self.is_running = False
        self.monitor = PerformanceMonitor()
        self.video_writer = None
        
        # 历史数据
        self.people_history = deque(maxlen=30)
        self.zone_history = deque(maxlen=30)
        
        # 统计
        self.stats = {
            'frames_processed': 0,
            'total_people_detected': 0,
            'ac_on_time': 0,
            'ac_off_time': 0,
            'ac_cycles': 0,
            'energy_estimate_kwh': 0,
            'start_time': None,
            'zone_stats': {}
        }
        
        # 初始化区域统计
        for zone in self.config['zones']:
            self.stats['zone_stats'][zone['name']] = {
                'total_people': 0,
                'peak_people': 0
            }
    
    def is_class_time(self) -> bool:
        """检查当前是否为上课时间"""
        if not self.config['time_policy']['enable']:
            return True
        
        now = datetime.now()
        current_time = now.strftime('%H:%M')
        
        for period in self.config['time_policy']['class_hours']:
            if period['start'] <= current_time <= period['end']:
                return True
        return False
    
    def get_target_temp(self) -> int:
        """根据时段获取目标温度"""
        hour = datetime.now().hour
        
        if 8 <= hour < 12:
            return self.config['time_policy']['temp_adjust'].get('morning', 26)
        elif 12 <= hour < 18:
            return self.config['time_policy']['temp_adjust'].get('afternoon', 25)
        else:
            return self.config['time_policy']['temp_adjust'].get('evening', 26)
    
    def start(self, source) -> bool:
        """启动系统"""
        logger.info("=" * 60)
        logger.info("教室空调智能控制系统 v2.0 启动")
        logger.info("=" * 60)
        logger.info(f"模式: {'Demo' if self.config['demo_mode'] else 'Deploy'}")
        logger.info(f"最少人数: {self.config['min_people']}")
        logger.info(f"冷却时间: {self.config['cooldown_minutes']}分钟")
        logger.info(f"时段策略: {'开启' if self.config['time_policy']['enable'] else '关闭'}")
        
        self.video = VideoCapture(source)
        if not self.video.start():
            logger.error("无法启动视频源")
            return False
        
        width, height = self.video.get_size()
        self.config['frame_size'] = (width, height)
        self.zone_manager.update_frame_size((width, height))
        
        logger.info(f"视频源: {source} | 分辨率: {width}x{height}")
        
        if self.config.get('save_video'):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.config['video_output'], fourcc, 10, (width, height)
            )
        
        if not self.counter.load_model():
            logger.error("模型加载失败")
            return False
        
        if not self.ac_controller.init():
            logger.error("控制器初始化失败")
            return False
        
        self.stats['start_time'] = time.time()
        self.is_running = True
        
        logger.info("系统初始化完成")
        logger.info("按 'q' 退出 | 's' 保存截图 | 'r' 查看报告")
        return True
    
    def analyze_crowd_density(self, detections: list, frame_shape: tuple) -> dict:
        """分析人群密度分布"""
        zones = self.zone_manager.get_zones()
        zone_counts = {name: 0 for name in zones.keys()}
        
        for det in detections:
            if det['class'] != 'person':
                continue
            x1, y1, x2, y2 = det['bbox']
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
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
            cv2.circle(heatmap, (cx, cy), 50, 1.0, -1)
        
        if heatmap.max() > 0:
            heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
            heatmap = heatmap / heatmap.max()
            heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
            result = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)
            return result
        
        return frame
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """处理单帧"""
        inference_start = time.time()
        
        # 1. 人形检测
        detections = self.counter.detect(frame)
        inference_time = (time.time() - inference_start) * 1000
        people_count = len([d for d in detections if d['class'] == 'person'])
        
        # 2. 区域分析
        zone_counts = self.analyze_crowd_density(detections, frame.shape)
        
        # 3. 更新历史
        self.people_history.append(people_count)
        self.zone_history.append(zone_counts)
        
        # 4. 平滑处理
        avg_people = np.mean(self.people_history) if self.people_history else 0
        
        # 5. 时段策略
        is_class_time = self.is_class_time()
        target_temp = self.get_target_temp() if is_class_time else self.config['target_temp_default']
        
        # 6. 可视化
        display_frame = self.draw_heatmap(frame, detections)
        
        # 绘制检测框
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            color = (0, 255, 0) if det['class'] == 'person' else (255, 0, 0)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        
        # 绘制区域
        self.zone_manager.draw_zones(display_frame)
        
        # 7. 空调控制决策
        if is_class_time:
            ac_state = self.ac_controller.update(avg_people)
        else:
            # 非上课时间自动关闭
            if self.ac_controller.is_on:
                self.ac_controller.turn_off()
            ac_state = False
        
        # 8. 更新统计
        self._update_stats(people_count, zone_counts, ac_state)
        
        # 9. 绘制信息面板
        self._draw_info_panel(display_frame, people_count, avg_people, zone_counts, 
                             ac_state, is_class_time, target_temp, inference_time)
        
        # 10. 性能监控
        self.monitor.record_frame(inference_time)
        if self.config.get('show_performance'):
            perf_lines = self.monitor.draw_overlay(display_frame)
            for i, line in enumerate(perf_lines):
                cv2.putText(display_frame, line, (display_frame.shape[1] - 200, 30 + i*25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        self.stats['frames_processed'] += 1
        self.stats['total_people_detected'] += people_count
        
        return display_frame
    
    def _update_stats(self, people_count, zone_counts, ac_state):
        """更新统计数据"""
        # 区域统计
        for zone_name, count in zone_counts.items():
            self.stats['zone_stats'][zone_name]['total_people'] += count
            self.stats['zone_stats'][zone_name]['peak_people'] = max(
                self.stats['zone_stats'][zone_name]['peak_people'], count
            )
        
        # 空调时间统计
        if ac_state:
            self.stats['ac_on_time'] += 1/15  # 假设15fps
        else:
            self.stats['ac_off_time'] += 1/15
        
        # 估算能耗 (假设1.5kW空调，粗略估计)
        if ac_state:
            self.stats['energy_estimate_kwh'] += 1.5 / 3600 / 15
    
    def _draw_info_panel(self, frame, current, avg, zones, ac_state, 
                        is_class_time, target_temp, inference_time):
        """绘制信息面板"""
        h, w = frame.shape[:2]
        
        # 背景
        panel_h = 180 + len(zones) * 25
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (380, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 文字
        y_offset = 35
        cv2.putText(frame, f"People: {current} | Avg: {avg:.1f}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        y_offset += 30
        time_status = "CLASS TIME" if is_class_time else "FREE TIME"
        cv2.putText(frame, f"{time_status} | Target: {target_temp}°C",
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        y_offset += 30
        for zone, count in zones.items():
            cv2.putText(frame, f"{zone}: {count} people", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 25
        
        y_offset += 10
        ac_color = (0, 255, 0) if ac_state else (0, 0, 255)
        ac_text = "AC: ON" if ac_state else "AC: OFF"
        cv2.putText(frame, ac_text, (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, ac_color, 2)
        
        y_offset += 30
        mode_text = "DEMO MODE" if self.config['demo_mode'] else "DEPLOY MODE"
        cv2.putText(frame, mode_text, (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
    
    def print_report(self):
        """打印节能报告"""
        if not self.stats['start_time']:
            return
        
        runtime = time.time() - self.stats['start_time']
        perf = self.monitor.get_summary()
        
        print("\n" + "=" * 60)
        print("节能运行报告")
        print("=" * 60)
        print(f"运行时间: {runtime/3600:.2f} 小时")
        print(f"总检测人次: {self.stats['total_people_detected']}")
        print(f"平均FPS: {perf['avg_fps']:.1f}")
        print(f"\n空调统计:")
        print(f"  运行时间: {self.stats['ac_on_time']/3600:.2f} 小时")
        print(f"  关闭时间: {self.stats['ac_off_time']/3600:.2f} 小时")
        print(f"  启停次数: {self.stats['ac_cycles']}")
        print(f"  估算能耗: {self.stats['energy_estimate_kwh']:.2f} kWh")
        
        if self.stats['ac_off_time'] > 0:
            saved_hours = self.stats['ac_off_time'] / 3600
            saved_kwh = saved_hours * 1.5  # 假设1.5kW
            print(f"\n节能估算:")
            print(f"  节省运行时间: {saved_hours:.2f} 小时")
            print(f"  估算节省电量: {saved_kwh:.2f} kWh")
        
        print(f"\n区域统计:")
        for zone, stats in self.stats['zone_stats'].items():
            print(f"  {zone}: 总计{stats['total_people']}人次, 峰值{stats['peak_people']}人")
        print("=" * 60 + "\n")
    
    def run(self):
        """主循环"""
        frame_interval = 1.0 / 10
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
            
            display_frame = self.process_frame(frame)
            
            if self.video_writer:
                self.video_writer.write(display_frame)
            
            cv2.imshow("Classroom AC Control", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_count += 1
                filename = f"classroom_screenshot_{screenshot_count:03d}.jpg"
                cv2.imwrite(filename, display_frame)
                logger.info(f"截图保存: {filename}")
            elif key == ord('r'):
                self.print_report()
        
        self.stop()
    
    def stop(self):
        """停止系统"""
        self.is_running = False
        
        # 打印最终报告
        self.print_report()
        
        if self.video:
            self.video.stop()
        if self.video_writer:
            self.video_writer.release()
        self.ac_controller.cleanup()
        cv2.destroyAllWindows()
        logger.info("系统已停止")


def main():
    parser = argparse.ArgumentParser(description='教室空调智能控制系统 v2.0')
    parser.add_argument('--source', type=str, default=None,
                       help='视频源: 0=摄像头, 路径=视频文件')
    parser.add_argument('--mode', type=str, default=None, choices=['demo', 'deploy'],
                       help='运行模式')
    parser.add_argument('--min-people', type=int, default=None,
                       help='开空调的最少人数')
    parser.add_argument('--cooldown', type=int, default=None,
                       help='空调关闭后等待分钟数才能再开')
    parser.add_argument('--config', type=str, default='classroom_ac/config.yaml',
                       help='配置文件路径')
    parser.add_argument('--save-video', action='store_true',
                       help='保存处理后的视频')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config, ClassroomACSystem.DEFAULT_CONFIG)
    config = merge_config(config, args)
    
    # 解析视频源
    source = config.get('source', 0)
    if args.source is not None:
        source = int(args.source) if args.source.isdigit() else args.source
    
    # 启动系统
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
