#!/usr/bin/env python3
"""
昼夜节律模块测试
"""

import unittest
import sys
from pathlib import Path
from datetime import datetime, time

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from circadian_rhythm import (
    CircadianRhythm, 
    ColorTemperatureConfig
)


class TestCircadianRhythm(unittest.TestCase):
    """测试昼夜节律控制器"""
    
    def setUp(self):
        """测试前准备"""
        self.circadian = CircadianRhythm()
    
    def test_morning_color_temperature(self):
        """测试早晨色温"""
        # 07:00 应该在早晨时段
        morning_time = datetime(2024, 1, 1, 7, 0, 0)
        ct = self.circadian.get_color_temperature(morning_time)
        self.assertGreaterEqual(ct, 5000)
        self.assertLessEqual(ct, 6500)
    
    def test_work_color_temperature(self):
        """测试工作时段色温"""
        # 12:00 应该在工作时段
        work_time = datetime(2024, 1, 1, 12, 0, 0)
        ct = self.circadian.get_color_temperature(work_time)
        self.assertGreaterEqual(ct, 4000)
        self.assertLessEqual(ct, 5000)
    
    def test_evening_color_temperature(self):
        """测试傍晚色温"""
        # 18:00 应该在傍晚时段
        evening_time = datetime(2024, 1, 1, 18, 0, 0)
        ct = self.circadian.get_color_temperature(evening_time)
        self.assertGreaterEqual(ct, 3000)
        self.assertLessEqual(ct, 4000)
    
    def test_night_color_temperature(self):
        """测试夜间色温"""
        # 22:00 应该在夜间时段
        night_time = datetime(2024, 1, 1, 22, 0, 0)
        ct = self.circadian.get_color_temperature(night_time)
        self.assertGreaterEqual(ct, 2700)
        self.assertLessEqual(ct, 3000)
    
    def test_morning_brightness(self):
        """测试早晨亮度"""
        morning_time = datetime(2024, 1, 1, 7, 0, 0)
        brightness = self.circadian.get_brightness(morning_time)
        self.assertAlmostEqual(brightness, 1.0, places=1)
    
    def test_night_brightness(self):
        """测试夜间亮度"""
        night_time = datetime(2024, 1, 1, 22, 0, 0)
        brightness = self.circadian.get_brightness(night_time)
        self.assertAlmostEqual(brightness, 0.4, places=1)
    
    def test_manual_override(self):
        """测试手动覆盖"""
        # 设置手动覆盖
        self.circadian.set_manual_override(color_temp=5000, brightness=0.8)
        
        # 验证手动设置生效
        ct, br = self.circadian.get_lighting_state()
        self.assertEqual(ct, 5000)
        self.assertEqual(br, 0.8)
        
        # 清除手动覆盖
        self.circadian.clear_manual_override()
        self.assertFalse(self.circadian.is_manual_override_active())
    
    def test_lighting_state(self):
        """测试获取照明状态"""
        morning_time = datetime(2024, 1, 1, 7, 0, 0)
        ct, brightness = self.circadian.get_lighting_state(morning_time)
        
        self.assertIsInstance(ct, int)
        self.assertIsInstance(brightness, float)
        self.assertGreaterEqual(ct, 2700)
        self.assertLessEqual(ct, 6500)
        self.assertGreaterEqual(brightness, 0.0)
        self.assertLessEqual(brightness, 1.0)
    
    def test_daily_schedule(self):
        """测试每日计划"""
        schedule = self.circadian.get_daily_schedule()
        self.assertEqual(len(schedule), 4)
        
        # 检查每个时间段
        for period, ct, brightness, desc in schedule:
            self.assertIsInstance(period, str)
            self.assertIsInstance(ct, int)
            self.assertIsInstance(brightness, float)
            self.assertIsInstance(desc, str)


class TestColorTemperatureConfig(unittest.TestCase):
    """测试色温配置"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = ColorTemperatureConfig()
        
        self.assertEqual(config.morning_ct, 6500)
        self.assertEqual(config.work_ct, 4500)
        self.assertEqual(config.evening_ct, 3500)
        self.assertEqual(config.night_ct, 2700)
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = ColorTemperatureConfig(
            morning_ct=6000,
            work_ct=4000,
            morning_brightness=0.95
        )
        
        self.assertEqual(config.morning_ct, 6000)
        self.assertEqual(config.work_ct, 4000)
        self.assertEqual(config.morning_brightness, 0.95)


if __name__ == '__main__':
    unittest.main()
