#!/usr/bin/env python3
"""
性能监控和统计模块
"""
import time
import psutil
from collections import deque
from typing import Dict, List, Optional, Any, Deque
from dataclasses import dataclass, field

# 默认历史记录大小
DEFAULT_HISTORY_SIZE = 30

# 默认绘制位置
DEFAULT_OVERLAY_X = 10
DEFAULT_OVERLAY_Y = 30


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
    fps_history: Deque[float] = field(default_factory=lambda: deque(maxlen=DEFAULT_HISTORY_SIZE))
    inference_history: Deque[float] = field(default_factory=lambda: deque(maxlen=DEFAULT_HISTORY_SIZE))


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, history_size: int = DEFAULT_HISTORY_SIZE) -> None:
        """初始化性能监控器
        
        Args:
            history_size: 历史记录大小
        """
        self.stats: PerformanceStats = PerformanceStats()
        self.history_size: int = history_size
        self._start_time: float = time.time()
        self._last_frame_time: float = time.time()
        self._frame_times: Deque[float] = deque(maxlen=history_size)
        self._process: psutil.Process = psutil.Process()
    
    def record_frame(self, inference_time_ms: float = 0.0) -> None:
        """记录一帧的处理
        
        Args:
            inference_time_ms: 推理时间（毫秒）
        """
        now = time.time()
        frame_time = now - self._last_frame_time
        self._last_frame_time = now
        
        self._frame_times.append(frame_time)
        self.stats.frame_count += 1
        
        # 计算FPS
        if len(self._frame_times) > 1:
            avg_frame_time = sum(self._frame_times) / len(self._frame_times)
            self.stats.fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        
        # 记录推理时间
        self.stats.inference_time_ms = inference_time_ms
        self.stats.fps_history.append(self.stats.fps)
        self.stats.inference_history.append(inference_time_ms)
    
    def record_drop(self) -> None:
        """记录丢帧"""
        self.stats.drop_count += 1
    
    def update_system_stats(self) -> None:
        """更新系统资源使用情况"""
        try:
            self.stats.cpu_percent = self._process.cpu_percent()
            self.stats.memory_mb = self._process.memory_info().rss / 1024 / 1024
        except (psutil.Error, OSError):
            # 忽略系统信息获取失败的情况
            pass
    
    def get_summary(self) -> Dict[str, Any]:
        """获取性能摘要
        
        Returns:
            包含性能统计信息的字典
        """
        runtime = time.time() - self._start_time
        
        avg_fps = 0.0
        if self.stats.fps_history:
            avg_fps = sum(self.stats.fps_history) / len(self.stats.fps_history)
        
        avg_inference_ms = 0.0
        if self.stats.inference_history:
            avg_inference_ms = sum(self.stats.inference_history) / len(self.stats.inference_history)
        
        return {
            'runtime_seconds': runtime,
            'total_frames': self.stats.frame_count,
            'avg_fps': avg_fps,
            'avg_inference_ms': avg_inference_ms,
            'drop_rate': self.stats.drop_count / max(self.stats.frame_count, 1),
            'cpu_percent': self.stats.cpu_percent,
            'memory_mb': self.stats.memory_mb
        }
    
    def draw_overlay(self, frame: Any, x: int = DEFAULT_OVERLAY_X, 
                     y: int = DEFAULT_OVERLAY_Y) -> List[str]:
        """在帧上绘制性能信息（返回文本列表，由调用者绘制）
        
        Args:
            frame: 视频帧
            x: 绘制起始X坐标
            y: 绘制起始Y坐标
            
        Returns:
            性能信息文本列表
        """
        self.update_system_stats()
        
        lines: List[str] = [
            f"FPS: {self.stats.fps:.1f}",
            f"Inference: {self.stats.inference_time_ms:.1f}ms",
            f"CPU: {self.stats.cpu_percent:.1f}%",
            f"MEM: {self.stats.memory_mb:.1f}MB"
        ]
        
        return lines
