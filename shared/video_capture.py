#!/usr/bin/env python3
"""
共享视频捕获模块
支持：摄像头/USB Camera/视频文件/RTSP流
"""
import cv2
import time
import threading
from typing import Optional, Callable, Union
from pathlib import Path

import numpy as np

# 常量定义
DEFAULT_START_TIMEOUT_SECONDS = 5.0  # 启动超时时间
FRAME_WAIT_INTERVAL = 0.01  # 帧等待间隔（秒）
DEFAULT_BUFFER_SIZE = 1  # 默认缓冲区大小


class VideoCapture:
    """线程安全的视频捕获封装"""
    
    def __init__(self, source: Union[int, str] = 0, buffer_size: int = 1):
        """
        Args:
            source: 0=默认摄像头, 路径=视频文件, url=RTSP流
            buffer_size: 帧缓冲区大小
        """
        self.source = source
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.frame: Optional[np.ndarray] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.fps = 0
        self.frame_count = 0
        
    def start(self) -> bool:
        """启动视频捕获"""
        if isinstance(self.source, (str, Path)) and not str(self.source).isdigit():
            # 视频文件或URL
            self.cap = cv2.VideoCapture(str(self.source))
        else:
            # 摄像头
            self.cap = cv2.VideoCapture(int(self.source))
            # 设置缓冲区大小减少延迟
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self.cap.isOpened():
            print(f"错误: 无法打开视频源 {self.source}")
            return False
        
        self.is_running = True
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()
        
        # 等待第一帧
        start = time.time()
        while self.frame is None and time.time() - start < DEFAULT_START_TIMEOUT_SECONDS:
            time.sleep(FRAME_WAIT_INTERVAL)
        
        return self.frame is not None
    
    def _update(self):
        """后台线程持续读取帧"""
        prev_time = time.time()
        fps_counter = 0
        fps_time = time.time()
        
        while self.is_running:
            if self.cap is None:
                break
                
            ret, frame = self.cap.read()
            
            if not ret:
                # 视频文件结束，循环播放（测试用）
                if isinstance(self.source, (str, Path)):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
            
            with self._lock:
                self.frame = frame
                self.frame_count += 1
            
            # 计算FPS
            fps_counter += 1
            if time.time() - fps_time >= 1.0:
                self.fps = fps_counter
                fps_counter = 0
                fps_time = time.time()
    
    def read(self) -> Optional[np.ndarray]:
        """获取最新帧（线程安全）"""
        with self._lock:
            return self.frame.copy() if self.frame is not None else None
    
    def get_size(self) -> tuple:
        """获取视频尺寸"""
        if self.cap:
            return (
                int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )
        return (0, 0)
    
    def stop(self):
        """停止捕获"""
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None


class FrameProcessor:
    """帧处理器基类"""
    
    def __init__(self, target_fps: int = 15):
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.last_process_time = 0
    
    def should_process(self) -> bool:
        """检查是否应该处理当前帧（限流）"""
        now = time.time()
        if now - self.last_process_time >= self.frame_interval:
            self.last_process_time = now
            return True
        return False
