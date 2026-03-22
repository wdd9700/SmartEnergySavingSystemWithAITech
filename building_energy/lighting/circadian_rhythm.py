#!/usr/bin/env python3
"""
昼夜节律照明控制器

实现基于时间的色温调节算法，符合人体昼夜节律。

色温调节策略:
- 06:00-09:00: 5000-6500K 冷白光，提神醒脑
- 09:00-17:00: 4000-5000K 自然白光，保持专注
- 17:00-20:00: 3000-4000K 暖白光，放松过渡
- 20:00-06:00: 2700-3000K 暖黄光，促进睡眠
"""

from typing import Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, time, timedelta
import math


@dataclass
class ColorTemperatureConfig:
    """色温配置"""
    morning_start: time = time(6, 0)      # 早晨开始
    morning_end: time = time(9, 0)        # 早晨结束
    work_start: time = time(9, 0)         # 工作时段开始
    work_end: time = time(17, 0)          # 工作时段结束
    evening_start: time = time(17, 0)     # 傍晚开始
    evening_end: time = time(20, 0)       # 傍晚结束
    night_start: time = time(20, 0)       # 夜间开始
    night_end: time = time(6, 0)          # 夜间结束
    
    # 色温值 (K)
    morning_ct: int = 6500                # 早晨色温
    work_ct: int = 4500                   # 工作时段色温
    evening_ct: int = 3500                # 傍晚色温
    night_ct: int = 2700                  # 夜间色温
    
    # 亮度比例 (0-1)
    morning_brightness: float = 1.0       # 早晨亮度
    work_brightness: float = 0.9          # 工作时段亮度
    evening_brightness: float = 0.7       # 傍晚亮度
    night_brightness: float = 0.4         # 夜间亮度
    
    # 过渡时间 (分钟)
    transition_minutes: int = 30          # 色温过渡时间


class CircadianRhythm:
    """
    昼夜节律照明控制器
    
    根据当前时间自动计算推荐的色温和亮度，符合人体昼夜节律。
    支持平滑过渡，避免色温突变。
    
    Attributes:
        config: 色温配置
        manual_override: 是否启用手动覆盖
        manual_color_temp: 手动设置的色温
        manual_brightness: 手动设置的亮度
    """
    
    def __init__(self, 
                 latitude: float = 39.9,    # 默认北京纬度
                 longitude: float = 116.4,  # 默认北京经度
                 config: Optional[ColorTemperatureConfig] = None):
        """
        初始化昼夜节律控制器
        
        Args:
            latitude: 纬度，用于计算日出日落时间
            longitude: 经度，用于计算日出日落时间
            config: 色温配置，None则使用默认配置
        """
        self.latitude = latitude
        self.longitude = longitude
        self.config = config or ColorTemperatureConfig()
        
        # 手动覆盖状态
        self._manual_override = False
        self._manual_color_temp: Optional[int] = None
        self._manual_brightness: Optional[float] = None
        self._manual_expire_time: Optional[datetime] = None
    
    def get_color_temperature(self, 
                              timestamp: Optional[datetime] = None) -> int:
        """
        获取当前推荐色温
        
        Args:
            timestamp: 指定时间，None则使用当前时间
            
        Returns:
            色温值 (K)，范围 2700-6500
        """
        # 检查手动覆盖
        if self._manual_override and self._is_manual_valid():
            return self._manual_color_temp or self._calculate_color_temperature(timestamp)
        
        return self._calculate_color_temperature(timestamp)
    
    def get_brightness(self, 
                       timestamp: Optional[datetime] = None) -> float:
        """
        获取当前推荐亮度
        
        Args:
            timestamp: 指定时间，None则使用当前时间
            
        Returns:
            亮度比例 (0-1)
        """
        # 检查手动覆盖
        if self._manual_override and self._is_manual_valid():
            return self._manual_brightness or self._calculate_brightness(timestamp)
        
        return self._calculate_brightness(timestamp)
    
    def get_lighting_state(self, 
                          timestamp: Optional[datetime] = None) -> Tuple[int, float]:
        """
        获取当前照明状态 (色温 + 亮度)
        
        Args:
            timestamp: 指定时间，None则使用当前时间
            
        Returns:
            (色温K, 亮度比例)
        """
        return (
            self.get_color_temperature(timestamp),
            self.get_brightness(timestamp)
        )
    
    def set_manual_override(self, 
                           color_temp: Optional[int] = None,
                           brightness: Optional[float] = None,
                           duration_minutes: int = 60):
        """
        设置手动覆盖
        
        Args:
            color_temp: 手动色温，None则保持自动
            brightness: 手动亮度，None则保持自动
            duration_minutes: 手动设置的有效期（分钟）
        """
        self._manual_override = True
        self._manual_color_temp = color_temp
        self._manual_brightness = brightness
        self._manual_expire_time = datetime.now() + timedelta(minutes=duration_minutes)
    
    def clear_manual_override(self):
        """清除手动覆盖，恢复自动控制"""
        self._manual_override = False
        self._manual_color_temp = None
        self._manual_brightness = None
        self._manual_expire_time = None
    
    def is_manual_override_active(self) -> bool:
        """检查是否处于手动覆盖状态"""
        return self._manual_override and self._is_manual_valid()
    
    def _is_manual_valid(self) -> bool:
        """检查手动设置是否仍在有效期内"""
        if self._manual_expire_time is None:
            return False
        return datetime.now() < self._manual_expire_time
    
    def _calculate_color_temperature(self, 
                                     timestamp: Optional[datetime] = None) -> int:
        """计算当前色温"""
        now = timestamp or datetime.now()
        current_time = now.time()
        cfg = self.config
        
        # 定义时间段
        periods = [
            (cfg.morning_start, cfg.morning_end, cfg.morning_ct),
            (cfg.work_start, cfg.work_end, cfg.work_ct),
            (cfg.evening_start, cfg.evening_end, cfg.evening_ct),
        ]
        
        # 检查是否在定义的时间段内
        for start, end, ct in periods:
            if start <= current_time < end:
                # 检查是否需要过渡
                transition_start = self._time_minus_minutes(end, cfg.transition_minutes)
                if current_time >= transition_start:
                    # 在过渡期内，计算插值
                    next_ct = self._get_next_period_ct(end)
                    progress = self._calculate_transition_progress(
                        current_time, transition_start, end
                    )
                    return int(ct + (next_ct - ct) * progress)
                return ct
        
        # 夜间时段 (包括跨午夜的情况)
        return cfg.night_ct
    
    def _calculate_brightness(self, 
                              timestamp: Optional[datetime] = None) -> float:
        """计算当前亮度"""
        now = timestamp or datetime.now()
        current_time = now.time()
        cfg = self.config
        
        # 定义时间段
        periods = [
            (cfg.morning_start, cfg.morning_end, cfg.morning_brightness),
            (cfg.work_start, cfg.work_end, cfg.work_brightness),
            (cfg.evening_start, cfg.evening_end, cfg.evening_brightness),
        ]
        
        # 检查是否在定义的时间段内
        for start, end, brightness in periods:
            if start <= current_time < end:
                # 检查是否需要过渡
                transition_start = self._time_minus_minutes(end, cfg.transition_minutes)
                if current_time >= transition_start:
                    # 在过渡期内，计算插值
                    next_brightness = self._get_next_period_brightness(end)
                    progress = self._calculate_transition_progress(
                        current_time, transition_start, end
                    )
                    return brightness + (next_brightness - brightness) * progress
                return brightness
        
        # 夜间时段
        return cfg.night_brightness
    
    def _get_next_period_ct(self, current_end: time) -> int:
        """获取下一个时间段的色温"""
        cfg = self.config
        if current_end == cfg.morning_end:
            return cfg.work_ct
        elif current_end == cfg.work_end:
            return cfg.evening_ct
        elif current_end == cfg.evening_end:
            return cfg.night_ct
        return cfg.morning_ct
    
    def _get_next_period_brightness(self, current_end: time) -> float:
        """获取下一个时间段的亮度"""
        cfg = self.config
        if current_end == cfg.morning_end:
            return cfg.work_brightness
        elif current_end == cfg.work_end:
            return cfg.evening_brightness
        elif current_end == cfg.evening_end:
            return cfg.night_brightness
        return cfg.morning_brightness
    
    def _time_minus_minutes(self, t: time, minutes: int) -> time:
        """时间减去分钟数"""
        dt = datetime.combine(datetime.today(), t) - timedelta(minutes=minutes)
        return dt.time()
    
    def _calculate_transition_progress(self, 
                                       current: time, 
                                       start: time, 
                                       end: time) -> float:
        """计算过渡进度 (0-1)"""
        current_seconds = current.hour * 3600 + current.minute * 60 + current.second
        start_seconds = start.hour * 3600 + start.minute * 60 + start.second
        end_seconds = end.hour * 3600 + end.minute * 60 + end.second
        
        if end_seconds <= start_seconds:
            end_seconds += 24 * 3600
        if current_seconds < start_seconds:
            current_seconds += 24 * 3600
        
        total_duration = end_seconds - start_seconds
        elapsed = current_seconds - start_seconds
        
        progress = elapsed / total_duration if total_duration > 0 else 0
        return max(0.0, min(1.0, progress))
    
    def get_daily_schedule(self) -> list:
        """
        获取每日照明计划
        
        Returns:
            时间段列表，每项包含 (开始时间, 结束时间, 色温, 亮度)
        """
        cfg = self.config
        return [
            ("06:00-09:00", cfg.morning_ct, cfg.morning_brightness, "冷白光，提神醒脑"),
            ("09:00-17:00", cfg.work_ct, cfg.work_brightness, "自然白光，保持专注"),
            ("17:00-20:00", cfg.evening_ct, cfg.evening_brightness, "暖白光，放松过渡"),
            ("20:00-06:00", cfg.night_ct, cfg.night_brightness, "暖黄光，促进睡眠"),
        ]
