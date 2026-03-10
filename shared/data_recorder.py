#!/usr/bin/env python3
"""
数据记录与分析模块
支持:
- 检测事件记录 (CSV格式)
- 实时数据统计
- 热力图生成
- 能耗估算
"""
import csv
import json
import time
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque, defaultdict
import threading


class DataRecorder:
    """数据记录器"""
    
    def __init__(self, 
                 log_dir: str = "logs",
                 max_memory_records: int = 10000,
                 csv_flush_interval: int = 60):
        """
        Args:
            log_dir: 日志目录
            max_memory_records: 内存中保留的最大记录数
            csv_flush_interval: CSV刷新间隔(秒)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_memory_records = max_memory_records
        self.csv_flush_interval = csv_flush_interval
        
        # 内存记录队列
        self.detection_records: deque = deque(maxlen=max_memory_records)
        self.event_records: deque = deque(maxlen=max_memory_records)
        
        # CSV文件
        self.csv_files: Dict[str, csv.writer] = {}
        self.csv_file_objects: Dict[str, object] = {}
        
        # 锁
        self._lock = threading.Lock()
        
        # 初始化CSV
        self._init_csv_files()
    
    def _init_csv_files(self):
        """初始化CSV文件"""
        today = datetime.now().strftime('%Y%m%d')
        
        # 检测记录
        detection_path = self.log_dir / f"detections_{today}.csv"
        detection_exists = detection_path.exists()
        
        self.csv_file_objects['detection'] = open(detection_path, 'a', newline='', encoding='utf-8')
        self.csv_files['detection'] = csv.writer(self.csv_file_objects['detection'])
        
        if not detection_exists:
            self.csv_files['detection'].writerow([
                'timestamp', 'event_type', 'camera_id', 'people_count',
                'light_states', 'brightness', 'inference_time_ms', 'fps'
            ])
        
        # 事件记录
        event_path = self.log_dir / f"events_{today}.csv"
        event_exists = event_path.exists()
        
        self.csv_file_objects['event'] = open(event_path, 'a', newline='', encoding='utf-8')
        self.csv_files['event'] = csv.writer(self.csv_file_objects['event'])
        
        if not event_exists:
            self.csv_files['event'].writerow([
                'timestamp', 'event_type', 'details', 'camera_id'
            ])
    
    def record_detection(self,
                         timestamp: datetime,
                         camera_id: str,
                         people_count: int,
                         light_states: Dict[str, bool],
                         brightness: float,
                         inference_time_ms: float,
                         fps: float,
                         person_locations: List[Tuple[int, int]] = None):
        """
        记录检测帧数据
        
        Args:
            timestamp: 时间戳
            camera_id: 摄像头ID
            people_count: 人数
            light_states: 灯光状态 {light_id: on/off}
            brightness: 画面亮度
            inference_time_ms: 推理耗时
            fps: 当前帧率
            person_locations: 人形位置列表 [(x, y), ...]
        """
        record = {
            'timestamp': timestamp.isoformat(),
            'camera_id': camera_id,
            'people_count': people_count,
            'light_states': json.dumps(light_states),
            'brightness': round(brightness, 2),
            'inference_time_ms': round(inference_time_ms, 2),
            'fps': round(fps, 2),
            'person_locations': json.dumps(person_locations) if person_locations else '[]'
        }
        
        with self._lock:
            self.detection_records.append(record)
            
            # 写入CSV
            self.csv_files['detection'].writerow([
                record['timestamp'],
                'detection',
                record['camera_id'],
                record['people_count'],
                record['light_states'],
                record['brightness'],
                record['inference_time_ms'],
                record['fps']
            ])
    
    def record_event(self,
                    event_type: str,
                    details: str,
                    camera_id: str = "main"):
        """
        记录事件
        
        Args:
            event_type: 事件类型 (light_on/light_off/person_enter/person_leave/alert)
            details: 事件详情
            camera_id: 摄像头ID
        """
        timestamp = datetime.now()
        
        record = {
            'timestamp': timestamp.isoformat(),
            'event_type': event_type,
            'details': details,
            'camera_id': camera_id
        }
        
        with self._lock:
            self.event_records.append(record)
            
            self.csv_files['event'].writerow([
                record['timestamp'],
                record['event_type'],
                record['details'],
                record['camera_id']
            ])
            
            # 立即刷新重要事件
            if event_type in ['alert', 'error']:
                self.csv_file_objects['event'].flush()
    
    def flush(self):
        """强制刷新所有CSV文件"""
        with self._lock:
            for fobj in self.csv_file_objects.values():
                fobj.flush()
    
    def close(self):
        """关闭所有文件"""
        self.flush()
        with self._lock:
            for fobj in self.csv_file_objects.values():
                fobj.close()
    
    def get_recent_detections(self, seconds: int = 300) -> List[Dict]:
        """获取最近N秒的检测记录"""
        cutoff = datetime.now() - timedelta(seconds=seconds)
        
        with self._lock:
            return [
                r for r in self.detection_records
                if datetime.fromisoformat(r['timestamp']) > cutoff
            ]
    
    def get_statistics(self, hours: int = 24) -> Dict:
        """获取统计数据"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            recent = [
                r for r in self.detection_records
                if datetime.fromisoformat(r['timestamp']) > cutoff
            ]
        
        if not recent:
            return {}
        
        people_counts = [r['people_count'] for r in recent]
        brightness_values = [r['brightness'] for r in recent]
        inference_times = [r['inference_time_ms'] for r in recent]
        
        return {
            'period_hours': hours,
            'total_records': len(recent),
            'people': {
                'avg': round(np.mean(people_counts), 2),
                'max': max(people_counts),
                'min': min(people_counts),
                'total_detections': sum(people_counts)
            },
            'brightness': {
                'avg': round(np.mean(brightness_values), 2),
                'min': min(brightness_values),
                'max': max(brightness_values)
            },
            'performance': {
                'avg_inference_ms': round(np.mean(inference_times), 2),
                'max_inference_ms': round(max(inference_times), 2),
                'min_fps': round(min(r['fps'] for r in recent), 2),
                'avg_fps': round(np.mean([r['fps'] for r in recent]), 2)
            }
        }
    
    def generate_hourly_report(self, hours: int = 24) -> Dict:
        """生成小时级统计报告"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            recent = [
                r for r in self.detection_records
                if datetime.fromisoformat(r['timestamp']) > cutoff
            ]
        
        if not recent:
            return {}
        
        # 按小时分组
        hourly = defaultdict(lambda: {'people_sum': 0, 'count': 0, 'light_on_time': 0})
        
        for r in recent:
            dt = datetime.fromisoformat(r['timestamp'])
            hour_key = dt.strftime('%Y-%m-%d %H:00')
            
            hourly[hour_key]['people_sum'] += r['people_count']
            hourly[hour_key]['count'] += 1
        
        # 计算每小时平均值
        report = {}
        for hour, data in sorted(hourly.items()):
            report[hour] = {
                'avg_people': round(data['people_sum'] / data['count'], 2) if data['count'] > 0 else 0,
                'detection_count': data['count']
            }
        
        return report
    
    def export_to_json(self, output_path: str, hours: int = 24):
        """导出记录到JSON"""
        data = {
            'export_time': datetime.now().isoformat(),
            'period_hours': hours,
            'statistics': self.get_statistics(hours),
            'hourly_report': self.generate_hourly_report(hours),
            'records': list(self.detection_records)[-1000:]  # 最近1000条
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return output_path


class HeatmapGenerator:
    """热力图生成器"""
    
    def __init__(self, frame_shape: Tuple[int, int]):
        """
        Args:
            frame_shape: (height, width)
        """
        self.frame_shape = frame_shape
        self.accumulator = np.zeros(frame_shape, dtype=np.float32)
        self.total_frames = 0
    
    def add_frame(self, person_locations: List[Tuple[int, int]], 
                  weight: float = 1.0):
        """
        添加一帧数据到热力图
        
        Args:
            person_locations: 人形位置列表 [(x, y), ...]
            weight: 权重
        """
        h, w = self.frame_shape
        
        for (px, py) in person_locations:
            # 限制在画面内
            px = max(0, min(px, w - 1))
            py = max(0, min(py, h - 1))
            
            # 使用高斯分布添加热力
            y_indices, x_indices = np.ogrid[:h, :w]
            gaussian = np.exp(-((x_indices - px)**2 + (y_indices - py)**2) / (2 * 50**2))
            self.accumulator += gaussian * weight
        
        self.total_frames += 1
    
    def generate(self, colormap: int = cv2.COLORMAP_JET, 
                 background: Optional[np.ndarray] = None) -> np.ndarray:
        """
        生成热力图
        
        Args:
            colormap: OpenCV颜色映射
            background: 背景图像 (可选)
        
        Returns:
            热力图图像
        """
        if self.total_frames == 0:
            return np.zeros((*self.frame_shape, 3), dtype=np.uint8)
        
        # 归一化
        normalized = self.accumulator / (self.accumulator.max() + 1e-8)
        normalized = (normalized * 255).astype(np.uint8)
        
        # 应用颜色映射
        heatmap = cv2.applyColorMap(normalized, colormap)
        
        if background is not None:
            # 叠加到背景
            heatmap = cv2.addWeighted(background, 0.6, heatmap, 0.4, 0)
        
        return heatmap
    
    def reset(self):
        """重置热力图"""
        self.accumulator.fill(0)
        self.total_frames = 0
    
    def save(self, filepath: str, background: Optional[np.ndarray] = None):
        """保存热力图"""
        heatmap = self.generate(background=background)
        cv2.imwrite(filepath, heatmap)


class EnergyEstimator:
    """能耗估算器"""
    
    # 假设功率 (瓦特)
    LIGHT_POWER = {
        'light_0': 20,  # 入口灯
        'light_1': 20,  # 中间灯
        'light_2': 20,  # 出口灯
    }
    
    def __init__(self):
        self.light_on_time: Dict[str, float] = defaultdict(float)
        self.light_last_on: Dict[str, Optional[datetime]] = {}
        self.total_energy_wh = 0.0
    
    def update_light_state(self, light_id: str, is_on: bool):
        """更新灯光状态"""
        now = datetime.now()
        
        if is_on and self.light_last_on.get(light_id) is None:
            # 灯刚开启
            self.light_last_on[light_id] = now
        elif not is_on and self.light_last_on.get(light_id) is not None:
            # 灯刚关闭，计算能耗
            on_duration = (now - self.light_last_on[light_id]).total_seconds() / 3600  # 小时
            power = self.LIGHT_POWER.get(light_id, 20)  # 默认20W
            energy = power * on_duration  # Wh
            
            self.light_on_time[light_id] += on_duration
            self.total_energy_wh += energy
            self.light_last_on[light_id] = None
    
    def get_statistics(self) -> Dict:
        """获取能耗统计"""
        # 计算当前开启的灯能耗
        now = datetime.now()
        current_energy = 0.0
        
        for light_id, last_on in self.light_last_on.items():
            if last_on is not None:
                duration = (now - last_on).total_seconds() / 3600
                power = self.LIGHT_POWER.get(light_id, 20)
                current_energy += power * duration
        
        return {
            'total_energy_wh': round(self.total_energy_wh + current_energy, 2),
            'total_energy_kwh': round((self.total_energy_wh + current_energy) / 1000, 4),
            'light_on_time_hours': {k: round(v, 2) for k, v in self.light_on_time.items()},
            'current_active': [k for k, v in self.light_last_on.items() if v is not None]
        }
    
    def estimate_savings(self, traditional_mode_hours: float) -> Dict:
        """
        估算节能效果
        
        Args:
            traditional_mode_hours: 传统模式下所有灯开启的小时数
        
        Returns:
            节能统计
        """
        actual = self.get_statistics()
        actual_kwh = actual['total_energy_kwh']
        
        # 传统模式能耗 (所有灯全开)
        total_power = sum(self.LIGHT_POWER.values())
        traditional_kwh = (total_power * traditional_mode_hours) / 1000
        
        savings_kwh = max(0, traditional_kwh - actual_kwh)
        savings_percent = (savings_kwh / traditional_kwh * 100) if traditional_kwh > 0 else 0
        
        return {
            'traditional_mode_kwh': round(traditional_kwh, 4),
            'smart_mode_kwh': round(actual_kwh, 4),
            'savings_kwh': round(savings_kwh, 4),
            'savings_percent': round(savings_percent, 2),
            'cost_savings_yuan': round(savings_kwh * 0.6, 2)  # 假设0.6元/度
        }
