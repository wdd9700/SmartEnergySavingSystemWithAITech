#!/usr/bin/env python3
"""
性能监控和统计模块
"""
import time
import psutil
from collections import deque
from typing import Dict
from dataclasses import dataclass, field


@dataclass
class PerformanceStats:
    """性能统计数据"""
    fps: float = 0.0
    inference_time_ms: float = 0.0
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    frame_count: int = 0
    drop_count: int = 0
    
    # 历史记录
    fps_history: deque = field(default_factory=lambda: deque(maxlen=30))
    inference_history: deque = field(default_factory=lambda: deque(maxlen=30))


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, history_size: int = 30):
        self.stats = PerformanceStats()
        self.history_size = history_size
        self._start_time = time.time()
        self._last_frame_time = time.time()
        self._frame_times = deque(maxlen=history_size)
        self._process = psutil.Process()
    
    def record_frame(self, inference_time_ms: float = 0):
        """记录一帧的处理"""
        now = time.time()
        frame_time = now - self._last_frame_time
        self._last_frame_time = now
        
        self._frame_times.append(frame_time)
        self.stats.frame_count += 1
        
        # 计算FPS
        if len(self._frame_times) > 1:
            avg_frame_time = sum(self._frame_times) / len(self._frame_times)
            self.stats.fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        # 记录推理时间
        self.stats.inference_time_ms = inference_time_ms
        self.stats.fps_history.append(self.stats.fps)
        self.stats.inference_history.append(inference_time_ms)
    
    def record_drop(self):
        """记录丢帧"""
        self.stats.drop_count += 1
    
    def update_system_stats(self):
        """更新系统资源使用情况"""
        try:
            self.stats.cpu_percent = self._process.cpu_percent()
            self.stats.memory_mb = self._process.memory_info().rss / 1024 / 1024
        except:
            pass
    
    def get_summary(self) -> Dict:
        """获取性能摘要"""
        runtime = time.time() - self._start_time
        
        return {
            'runtime_seconds': runtime,
            'total_frames': self.stats.frame_count,
            'avg_fps': sum(self.stats.fps_history) / len(self.stats.fps_history) if self.stats.fps_history else 0,
            'avg_inference_ms': sum(self.stats.inference_history) / len(self.stats.inference_history) if self.stats.inference_history else 0,
            'drop_rate': self.stats.drop_count / max(self.stats.frame_count, 1),
            'cpu_percent': self.stats.cpu_percent,
            'memory_mb': self.stats.memory_mb
        }
    
    def draw_overlay(self, frame, x=10, y=30):
        """在帧上绘制性能信息（返回文本列表，由调用者绘制）"""
        self.update_system_stats()
        
        lines = [
            f"FPS: {self.stats.fps:.1f}",
            f"Inference: {self.stats.inference_time_ms:.1f}ms",
            f"CPU: {self.stats.cpu_percent:.1f}%",
            f"MEM: {self.stats.memory_mb:.1f}MB"
        ]
        
        return lines
