#!/usr/bin/env python3
"""
电网压力计算器测试

测试GridPressureCalculator和相关数据类的功能。
"""

import unittest
from datetime import datetime, timedelta
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grid_calculator import (
    GridPressureCalculator,
    GridState,
    GridEvent,
    GridDataSimulator,
    GridStatus
)


class TestGridState(unittest.TestCase):
    """测试GridState类"""
    
    def test_create_grid_state(self):
        """测试创建GridState"""
        now = datetime.now()
        state = GridState(
            timestamp=now,
            voltage=220.0,
            frequency=50.0,
            load_factor=0.5,
            voltage_deviation=0.0,
            frequency_deviation=0.0,
            pressure_index=0.15,
            status="normal"
        )
        
        self.assertEqual(state.voltage, 220.0)
        self.assertEqual(state.frequency, 50.0)
        self.assertEqual(state.load_factor, 0.5)
        self.assertEqual(state.status, "normal")
    
    def test_to_dict(self):
        """测试转换为字典"""
        now = datetime.now()
        state = GridState(
            timestamp=now,
            voltage=220.0,
            frequency=50.0,
            load_factor=0.5
        )
        
        data = state.to_dict()
        self.assertIn("timestamp", data)
        self.assertEqual(data["voltage"], 220.0)
        self.assertEqual(data["status"], "normal")
    
    def test_from_dict(self):
        """测试从字典创建"""
        now = datetime.now()
        data = {
            "timestamp": now.isoformat(),
            "voltage": 220.0,
            "frequency": 50.0,
            "load_factor": 0.5,
            "voltage_deviation": 0.05,
            "frequency_deviation": 0.02,
            "pressure_index": 0.2,
            "status": "normal"
        }
        
        state = GridState.from_dict(data)
        self.assertEqual(state.voltage, 220.0)
        self.assertEqual(state.pressure_index, 0.2)


class TestGridPressureCalculator(unittest.TestCase):
    """测试GridPressureCalculator类"""
    
    def setUp(self):
        """测试前准备"""
        self.calculator = GridPressureCalculator()
    
    def test_normal_state(self):
        """测试正常状态计算"""
        state = self.calculator.calculate(
            voltage=220.0,
            frequency=50.0,
            load_factor=0.5
        )
        
        self.assertEqual(state.status, "normal")
        self.assertLess(state.pressure_index, 0.3)
        self.assertEqual(state.voltage, 220.0)
        self.assertEqual(state.frequency, 50.0)
    
    def test_warning_state(self):
        """测试警告状态计算"""
        # 电压偏低 + 负载较高 (需要压力指数 >= 0.3)
        state = self.calculator.calculate(
            voltage=190.0,
            frequency=48.5,
            load_factor=0.85
        )
        
        self.assertEqual(state.status, "warning")
        self.assertGreaterEqual(state.pressure_index, 0.3)
        self.assertLess(state.pressure_index, 0.6)
    
    def test_critical_state(self):
        """测试紧急状态计算"""
        # 电压很低 + 频率偏差大 + 负载很高 (需要压力指数 >= 0.6)
        state = self.calculator.calculate(
            voltage=90.0,
            frequency=38.0,
            load_factor=1.0
        )
        
        self.assertEqual(state.status, "critical")
        self.assertGreaterEqual(state.pressure_index, 0.6)
    
    def test_voltage_deviation_calculation(self):
        """测试电压偏差计算"""
        state = self.calculator.calculate(
            voltage=210.0,  # 低于标称220V
            frequency=50.0,
            load_factor=0.5
        )
        
        # 电压偏差 = |210-220|/220 = 4.55%
        expected_dev = abs(210 - 220) / 220 * 100
        self.assertAlmostEqual(state.voltage_deviation, expected_dev, places=1)
    
    def test_frequency_deviation_calculation(self):
        """测试频率偏差计算"""
        state = self.calculator.calculate(
            voltage=220.0,
            frequency=49.0,  # 低于标称50Hz
            load_factor=0.5
        )
        
        # 频率偏差 = |49-50|/50 = 2%
        expected_dev = abs(49 - 50) / 50 * 100
        self.assertAlmostEqual(state.frequency_deviation, expected_dev, places=1)
    
    def test_calculate_batch(self):
        """测试批量计算"""
        measurements = [
            {"voltage": 220.0, "frequency": 50.0, "load_factor": 0.5},
            {"voltage": 190.0, "frequency": 48.5, "load_factor": 0.85},
            {"voltage": 90.0, "frequency": 38.0, "load_factor": 1.0}
        ]
        
        states = self.calculator.calculate_batch(measurements)
        
        self.assertEqual(len(states), 3)
        self.assertEqual(states[0].status, "normal")
        self.assertEqual(states[1].status, "warning")
        self.assertEqual(states[2].status, "critical")
    
    def test_detect_events_high_load(self):
        """测试高负载事件检测"""
        state = self.calculator.calculate(
            voltage=220.0,
            frequency=50.0,
            load_factor=0.85
        )
        
        events = self.calculator.detect_events(state)
        
        high_load_events = [e for e in events if e.event_type == "high_load"]
        self.assertGreaterEqual(len(high_load_events), 1)
        self.assertEqual(high_load_events[0].severity, "medium")
    
    def test_detect_events_voltage_drop(self):
        """测试电压跌落事件检测"""
        state = self.calculator.calculate(
            voltage=200.0,  # 低于容差下限
            frequency=50.0,
            load_factor=0.5
        )
        
        events = self.calculator.detect_events(state)
        
        voltage_events = [e for e in events if e.event_type == "voltage_drop"]
        self.assertGreaterEqual(len(voltage_events), 1)
    
    def test_detect_events_frequency_deviation(self):
        """测试频率偏差事件检测"""
        state = self.calculator.calculate(
            voltage=220.0,
            frequency=49.0,  # 超出容差
            load_factor=0.5
        )
        
        events = self.calculator.detect_events(state)
        
        freq_events = [e for e in events if e.event_type == "frequency_deviation"]
        self.assertGreaterEqual(len(freq_events), 1)
    
    def test_get_pressure_trend_increasing(self):
        """测试压力上升趋势"""
        states = [
            GridState(timestamp=datetime.now(), voltage=220.0, frequency=50.0, 
                     load_factor=0.3, pressure_index=0.15, status="normal"),
            GridState(timestamp=datetime.now(), voltage=218.0, frequency=49.8, 
                     load_factor=0.4, pressure_index=0.25, status="normal"),
            GridState(timestamp=datetime.now(), voltage=215.0, frequency=49.5, 
                     load_factor=0.5, pressure_index=0.35, status="warning"),
            GridState(timestamp=datetime.now(), voltage=210.0, frequency=49.2, 
                     load_factor=0.6, pressure_index=0.45, status="warning"),
            GridState(timestamp=datetime.now(), voltage=205.0, frequency=49.0, 
                     load_factor=0.7, pressure_index=0.55, status="critical"),
        ]
        
        trend = self.calculator.get_pressure_trend(states)
        
        self.assertEqual(trend["trend"], "increasing")
        self.assertGreater(trend["slope"], 0)
    
    def test_get_pressure_trend_decreasing(self):
        """测试压力下降趋势"""
        states = [
            GridState(timestamp=datetime.now(), voltage=205.0, frequency=49.0, 
                     load_factor=0.7, pressure_index=0.55, status="critical"),
            GridState(timestamp=datetime.now(), voltage=210.0, frequency=49.2, 
                     load_factor=0.6, pressure_index=0.45, status="warning"),
            GridState(timestamp=datetime.now(), voltage=215.0, frequency=49.5, 
                     load_factor=0.5, pressure_index=0.35, status="warning"),
            GridState(timestamp=datetime.now(), voltage=218.0, frequency=49.8, 
                     load_factor=0.4, pressure_index=0.25, status="normal"),
            GridState(timestamp=datetime.now(), voltage=220.0, frequency=50.0, 
                     load_factor=0.3, pressure_index=0.15, status="normal"),
        ]
        
        trend = self.calculator.get_pressure_trend(states)
        
        self.assertEqual(trend["trend"], "decreasing")
        self.assertLess(trend["slope"], 0)
    
    def test_calibrate_thresholds(self):
        """测试阈值校准"""
        states = [
            GridState(timestamp=datetime.now(), voltage=220.0, frequency=50.0, 
                     load_factor=0.3, pressure_index=0.1, status="normal"),
            GridState(timestamp=datetime.now(), voltage=220.0, frequency=50.0, 
                     load_factor=0.4, pressure_index=0.2, status="normal"),
            GridState(timestamp=datetime.now(), voltage=220.0, frequency=50.0, 
                     load_factor=0.5, pressure_index=0.3, status="warning"),
            GridState(timestamp=datetime.now(), voltage=220.0, frequency=50.0, 
                     load_factor=0.6, pressure_index=0.4, status="warning"),
            GridState(timestamp=datetime.now(), voltage=220.0, frequency=50.0, 
                     load_factor=0.7, pressure_index=0.5, status="critical"),
        ]
        
        thresholds = self.calculator.calibrate_thresholds(states, target_normal_ratio=0.6)
        
        self.assertIn("warning", thresholds)
        self.assertIn("critical", thresholds)
        self.assertLess(thresholds["warning"], thresholds["critical"])


class TestGridDataSimulator(unittest.TestCase):
    """测试GridDataSimulator类"""
    
    def setUp(self):
        """测试前准备"""
        self.simulator = GridDataSimulator()
    
    def test_generate(self):
        """测试数据生成"""
        data = self.simulator.generate()
        
        self.assertIn("voltage", data)
        self.assertIn("frequency", data)
        self.assertIn("load_factor", data)
        
        self.assertGreater(data["voltage"], 200)
        self.assertLess(data["voltage"], 240)
        self.assertGreater(data["frequency"], 48)
        self.assertLess(data["frequency"], 52)
        self.assertGreaterEqual(data["load_factor"], 0)
        self.assertLessEqual(data["load_factor"], 1)
    
    def test_generate_batch(self):
        """测试批量生成"""
        batch = self.simulator.generate_batch(10)
        
        self.assertEqual(len(batch), 10)
        
        for data in batch:
            self.assertIn("voltage", data)
            self.assertIn("frequency", data)
            self.assertIn("load_factor", data)


if __name__ == "__main__":
    unittest.main()
