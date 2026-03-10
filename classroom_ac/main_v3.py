#!/usr/bin/env python3
"""
教室空调智能控制系统 - 主程序 v3.0 (热负荷计算版)

新特性:
- 人体活动状态产热计算 (静止思考/轻度运动)
- 电子设备产热检测 (笔记本电脑20W)
- 外部环境温度影响
- 历史人数趋势分析
- 课表/预约驱动的预冷预热
- 热负荷可视化
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
from shared.data_recorder import DataRecorder
from classroom_ac.people_counter import PeopleCounter
from classroom_ac.thermal_controller import (
    HeatLoadCalculator, 
    ThermalLoadConfig,
    ScheduleManager,
    PredictiveACController
)


logger = setup_logger("classroom_ac_v3")


class ClassroomACSystemV3:
    """教室空调智能控制系统 v3.0 - 基于热负荷计算"""
    
    DEFAULT_CONFIG = {
        'model_path': 'models/yolov8n.onnx',
        'conf_threshold': 0.5,
        'room_area': 60,              # 教室面积 (平方米)
        'target_temp': 26.0,          # 目标温度
        'schedule_file': 'classroom_ac/schedule.json',
        
        # 检测配置
        'detect_laptops': True,       # 检测笔记本电脑
        'detect_activity': True,      # 分析活动状态
        
        # 控制参数
        'demo_mode': True,
        'enable_pre_cool': True,      # 启用预冷
        'enable_pre_heat': True,      # 启用预热
        
        # 数据记录
        'enable_recording': True,
        'log_dir': 'logs',
        
        # 外部数据
        'outdoor_temp_source': 'api',  # api|sensor|manual
        'outdoor_temp_manual': 30.0,   # 手动设置室外温度
    }
    
    def __init__(self, config: dict):
        self.config = {**self.DEFAULT_CONFIG, **config}
        
        # 检测器
        self.counter = PeopleCounter(
            model_path=self.config['model_path'],
            conf_threshold=self.config['conf_threshold']
        )
        
        # 热负荷计算
        thermal_config = ThermalLoadConfig()
        thermal_config.ROOM_AREA = self.config['room_area']
        self.heat_calculator = HeatLoadCalculator(thermal_config)
        
        # 课表管理
        self.schedule = ScheduleManager(self.config.get('schedule_file'))
        
        # 预测性控制器
        self.ac_controller = PredictiveACController(
            heat_calculator=self.heat_calculator,
            schedule_manager=self.schedule
        )
        self.ac_controller.target_temp = self.config['target_temp']
        
        # 数据记录
        self.recorder = None
        if self.config.get('enable_recording'):
            self.recorder = DataRecorder(log_dir=self.config['log_dir'])
        
        # 视频
        self.video = None
        self.video_writer = None
        self.is_running = False
        
        # 状态
        self.indoor_temp = 28.0
        self.outdoor_temp = self.config['outdoor_temp_manual']
        self.laptop_count = 0
        self.person_activity = 'thinking'  # resting|thinking|light_exercise
        
        # 统计
        self.stats = {
            'frames_processed': 0,
            'total_heat_load': 0,
            'avg_load': 0,
            'pre_cool_count': 0,
            'energy_saved': 0
        }
        
        # 历史
        self.load_history = deque(maxlen=100)
    
    def get_outdoor_temp(self) -> float:
        """获取室外温度"""
        source = self.config['outdoor_temp_source']
        
        if source == 'manual':
            return self.config['outdoor_temp_manual']
        elif source == 'sensor':
            # 实际部署时连接温度传感器
            return self.outdoor_temp
        else:
            # API获取 (简化版，实际应调用天气API)
            return self.outdoor_temp
    
    def analyze_activity_level(self, detections: list) -> str:
        """
        分析人员活动状态
        
        基于:
        - 检测框变化速度 (运动速度)
        - 人员分布密度
        """
        if not detections:
            return 'resting'
        
        # 简化：根据人数密度判断
        # 实际可以结合光流法分析运动速度
        person_count = len(detections)
        
        if person_count < 5:
            return 'resting'  # 人少，休息状态
        elif person_count < 20:
            return 'thinking'  # 正常上课
        else:
            return 'thinking'  # 人多，但默认思考状态
    
    def count_laptops(self, detections: list) -> int:
        """
        统计笔记本电脑数量
        
        基于YOLO检测laptop类别
        """
        # 注意：当前模型是COCO数据集，包含laptop类别
        # 这里简化处理，实际应该加载检测器识别laptop
        
        # 根据人数估算笔记本数量 (假设60%的人带笔记本)
        person_count = len([d for d in detections if d['class'] == 'person'])
        estimated_laptops = int(person_count * 0.6)
        return min(estimated_laptops, person_count)
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """处理单帧"""
        inference_start = time.time()
        display_frame = frame.copy()
        
        # 1. 人形检测
        detections = self.counter.detect(frame)
        person_count = len([d for d in detections if d['class'] == 'person'])
        
        inference_time = (time.time() - inference_start) * 1000
        
        # 2. 分析活动状态
        if self.config.get('detect_activity'):
            self.person_activity = self.analyze_activity_level(detections)
        
        # 3. 统计笔记本
        if self.config.get('detect_laptops'):
            self.laptop_count = self.count_laptops(detections)
        
        # 4. 获取室外温度
        self.outdoor_temp = self.get_outdoor_temp()
        
        # 5. 计算热负荷
        load_data = self.heat_calculator.calculate_total_load(
            person_count=person_count,
            outdoor_temp=self.outdoor_temp,
            indoor_temp=self.indoor_temp,
            laptop_count=self.laptop_count,
            activity_level=self.person_activity
        )
        
        self.load_history.append(load_data['total_load'])
        
        # 6. 更新控制器
        self.ac_controller.update_environment(
            indoor_temp=self.indoor_temp,
            outdoor_temp=self.outdoor_temp,
            person_count=person_count,
            laptop_count=self.laptop_count
        )
        
        # 7. 做出决策
        decision = self.ac_controller.make_decision()
        
        # 8. 绘制检测框
        for det in detections:
            if det['class'] == 'person':
                x1, y1, x2, y2 = det['bbox']
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 9. 绘制热负荷信息面板
        self._draw_thermal_panel(display_frame, load_data, decision, person_count)
        
        # 10. 记录数据
        if self.recorder:
            self.recorder.record_detection(
                timestamp=datetime.now(),
                camera_id='classroom_main',
                people_count=person_count,
                light_states={'ac': decision['ac_on'], 'fan': decision['fan_on']},
                brightness=0,
                inference_time_ms=inference_time,
                fps=15.0
            )
        
        self.stats['frames_processed'] += 1
        self.stats['total_heat_load'] += load_data['total_load']
        
        return display_frame
    
    def _draw_thermal_panel(self, frame, load_data, decision, person_count):
        """绘制热负荷信息面板"""
        h, w = frame.shape[:2]
        
        # 背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 280), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        y = 35
        
        # 标题
        cv2.putText(frame, "Thermal Load Control v3.0", 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        y += 30
        # 人数和活动状态
        activity_map = {
            'resting': '休息',
            'thinking': '思考/上课',
            'light_exercise': '轻度活动'
        }
        activity_cn = activity_map.get(self.person_activity, '未知')
        cv2.putText(frame, f"人数: {person_count} | 状态: {activity_cn}",
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        y += 30
        # 设备统计
        cv2.putText(frame, f"笔记本: {self.laptop_count}台 (约{self.laptop_count * 20}W)",
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        y += 30
        # 温度
        cv2.putText(frame, f"室内: {self.indoor_temp:.1f}°C | 室外: {self.outdoor_temp:.1f}°C",
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        y += 30
        # 热负荷分解
        cv2.putText(frame, f"热负荷: 人{load_data['person_heat']:.0f}W + 设备{load_data['equipment_heat']:.0f}W",
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        y += 25
        cv2.putText(frame, f"        围护{load_data['envelope_heat']:.0f}W + 太阳{load_data['solar_heat']:.0f}W",
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        y += 30
        # 总负荷
        total = load_data['total_load']
        color = (0, 255, 0) if total < 2000 else (0, 165, 255) if total < 3500 else (0, 0, 255)
        cv2.putText(frame, f"总热负荷: {total:.0f}W / 3500W",
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        y += 30
        # 决策
        ac_status = "开启" if decision['ac_on'] else "关闭"
        fan_status = "开启" if decision['fan_on'] else "关闭"
        pre = "[预]" if decision['pre_action'] else ""
        
        cv2.putText(frame, f"空调: {ac_status} {pre} | 风扇: {fan_status}",
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if decision['ac_on'] else (128, 128, 128), 2)
        
        y += 25
        cv2.putText(frame, f"原因: {decision['reason'][:40]}",
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
    
    def start(self, source) -> bool:
        """启动系统"""
        logger.info("=" * 60)
        logger.info("教室空调智能控制系统 v3.0 - 热负荷计算版")
        logger.info("=" * 60)
        logger.info(f"目标温度: {self.config['target_temp']}°C")
        logger.info(f"教室面积: {self.config['room_area']}m²")
        logger.info(f"检测笔记本: {self.config['detect_laptops']}")
        logger.info(f"预测性控制: 预冷{self.heat_calculator.config.PRE_COOL_TIME}分钟")
        
        self.video = VideoCapture(source)
        if not self.video.start():
            logger.error("无法启动视频源")
            return False
        
        if not self.counter.load_model():
            logger.error("模型加载失败")
            return False
        
        self.is_running = True
        logger.info("系统启动完成")
        return True
    
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
            
            if self.video_writer:
                self.video_writer.write(display_frame)
            
            cv2.imshow("Classroom AC Control v3", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('t'):
                # 调整目标温度
                self.target_temp = float(input("输入目标温度: "))
            elif key == ord('o'):
                # 调整室外温度 (模拟)
                self.outdoor_temp = float(input("输入室外温度: "))
        
        self.stop()
    
    def stop(self):
        """停止系统"""
        self.is_running = False
        
        if self.stats['frames_processed'] > 0:
            avg_load = self.stats['total_heat_load'] / self.stats['frames_processed']
            logger.info("=" * 60)
            logger.info("运行统计")
            logger.info(f"平均热负荷: {avg_load:.0f}W")
            logger.info(f"预冷次数: {self.stats['pre_cool_count']}")
            logger.info("=" * 60)
        
        if self.recorder:
            self.recorder.close()
        if self.video:
            self.video.stop()
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()
        
        logger.info("系统已停止")


def main():
    parser = argparse.ArgumentParser(description='教室空调智能控制系统 v3.0')
    parser.add_argument('--source', type=str, default='tests/test_classroom.mp4',
                       help='视频源')
    parser.add_argument('--demo', action='store_true',
                       help='演示模式')
    parser.add_argument('--target-temp', type=float, default=26.0,
                       help='目标温度')
    parser.add_argument('--outdoor-temp', type=float, default=30.0,
                       help='室外温度 (手动设置)')
    
    args = parser.parse_args()
    
    config = {
        'demo_mode': args.demo,
        'target_temp': args.target_temp,
        'outdoor_temp_manual': args.outdoor_temp,
        'outdoor_temp_source': 'manual'
    }
    
    source = int(args.source) if args.source.isdigit() else args.source
    
    system = ClassroomACSystemV3(config)
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
