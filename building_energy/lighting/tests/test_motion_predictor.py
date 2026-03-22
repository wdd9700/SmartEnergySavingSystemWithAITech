#!/usr/bin/env python3
"""
运动预测模块测试
"""

import unittest
import sys
from pathlib import Path
from datetime import datetime

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from motion_predictor import (
    MotionPredictor,
    MotionEvent,
    ZoneConfig,
    ZoneLayout
)


class TestZoneLayout(unittest.TestCase):
    """测试区域布局"""
    
    def setUp(self):
        """测试前准备"""
        self.layout = ZoneLayout()
        
        # 添加测试区域
        self.zone1 = ZoneConfig(
            id="zone_1",
            name="区域1",
            center=(100, 100),
            radius=50,
            neighbors={"east": "zone_2"}
        )
        self.zone2 = ZoneConfig(
            id="zone_2",
            name="区域2",
            center=(250, 100),
            radius=50,
            neighbors={"west": "zone_1", "east": "zone_3"}
        )
        self.zone3 = ZoneConfig(
            id="zone_3",
            name="区域3",
            center=(400, 100),
            radius=50,
            neighbors={"west": "zone_2"}
        )
        
        self.layout.add_zone(self.zone1)
        self.layout.add_zone(self.zone2)
        self.layout.add_zone(self.zone3)
    
    def test_add_zone(self):
        """测试添加区域"""
        self.assertEqual(len(self.layout.zones), 3)
        self.assertIsNotNone(self.layout.get_zone("zone_1"))
    
    def test_get_neighbors(self):
        """测试获取相邻区域"""
        neighbors = self.layout.get_neighbors("zone_1")
        self.assertIn("zone_2", neighbors)
        
        neighbors = self.layout.get_neighbors("zone_2")
        self.assertIn("zone_1", neighbors)
        self.assertIn("zone_3", neighbors)
    
    def test_calculate_distance(self):
        """测试计算距离"""
        distance = self.layout.calculate_distance("zone_1", "zone_2")
        self.assertAlmostEqual(distance, 150.0, places=0)
    
    def test_find_zone_by_position(self):
        """测试根据位置查找区域"""
        zone_id = self.layout.find_zone_by_position((100, 100))
        self.assertEqual(zone_id, "zone_1")
        
        zone_id = self.layout.find_zone_by_position((250, 100))
        self.assertEqual(zone_id, "zone_2")
    
    def test_get_direction_to_zone(self):
        """测试获取方向"""
        # zone_1 到 zone_2 应该是向东 (0度)
        direction = self.layout.get_direction_to_zone("zone_1", "zone_2")
        self.assertAlmostEqual(direction, 0.0, places=0)
        
        # zone_2 到 zone_1 应该是向西 (180度)
        direction = self.layout.get_direction_to_zone("zone_2", "zone_1")
        self.assertAlmostEqual(direction, 180.0, places=0)


class TestMotionPredictor(unittest.TestCase):
    """测试运动预测器"""
    
    def setUp(self):
        """测试前准备"""
        self.layout = ZoneLayout()
        
        # 添加测试区域
        self.zone1 = ZoneConfig(
            id="zone_1",
            name="区域1",
            center=(100, 100),
            radius=50,
            neighbors={"east": "zone_2"}
        )
        self.zone2 = ZoneConfig(
            id="zone_2",
            name="区域2",
            center=(250, 100),
            radius=50,
            neighbors={"west": "zone_1", "east": "zone_3"}
        )
        self.zone3 = ZoneConfig(
            id="zone_3",
            name="区域3",
            center=(400, 100),
            radius=50,
            neighbors={"west": "zone_2"}
        )
        
        self.layout.add_zone(self.zone1)
        self.layout.add_zone(self.zone2)
        self.layout.add_zone(self.zone3)
        
        self.predictor = MotionPredictor(self.layout)
    
    def test_update(self):
        """测试更新运动事件"""
        event = MotionEvent(
            timestamp=datetime.now(),
            zone_id="zone_1",
            position=(100, 100),
            direction=0.0,
            speed=1.0,
            confidence=0.9,
            track_id="person_1"
        )
        
        self.predictor.update(event)
        self.assertIn("person_1", self.predictor.history)
        self.assertEqual(len(self.predictor.history["person_1"]), 1)
    
    def test_predict_next_zone(self):
        """测试预测下一个区域"""
        # 向东运动，从zone_1应该预测到zone_2
        next_zone = self.predictor.predict_next_zone(
            current_zone="zone_1",
            direction=0.0
        )
        self.assertEqual(next_zone, "zone_2")
        
        # 向西运动，从zone_2应该预测到zone_1
        next_zone = self.predictor.predict_next_zone(
            current_zone="zone_2",
            direction=180.0
        )
        self.assertEqual(next_zone, "zone_1")
    
    def test_estimate_arrival_time(self):
        """测试估算到达时间"""
        # 距离150像素，速度1.2m/s，像素比例100像素/米
        # 距离 = 1.5米，时间 = 1.5/1.2 = 1.25秒
        arrival_time = self.predictor.estimate_arrival_time(
            from_zone="zone_1",
            to_zone="zone_2",
            current_speed=1.2
        )
        self.assertAlmostEqual(arrival_time, 1.25, places=1)
    
    def test_predict_destination(self):
        """测试预测目的地"""
        event = MotionEvent(
            timestamp=datetime.now(),
            zone_id="zone_1",
            position=(100, 100),
            direction=0.0,
            speed=1.2,
            confidence=0.9,
            track_id="person_1"
        )
        
        predictions = self.predictor.predict_destination(event)
        self.assertEqual(len(predictions), 1)
        
        zone_id, confidence, arrival_time = predictions[0]
        self.assertEqual(zone_id, "zone_2")
        self.assertGreater(confidence, 0.0)
        self.assertGreater(arrival_time, 0.0)
    
    def test_get_motion_trend(self):
        """测试获取运动趋势"""
        # 添加多个历史事件
        for i in range(5):
            event = MotionEvent(
                timestamp=datetime.now(),
                zone_id="zone_1",
                position=(100 + i * 10, 100),
                direction=0.0,
                speed=1.0,
                confidence=0.9,
                track_id="person_1"
            )
            self.predictor.update(event)
        
        trend = self.predictor.get_motion_trend("person_1")
        self.assertIsNotNone(trend)
        self.assertIn("avg_direction", trend)
        self.assertIn("avg_speed", trend)
        self.assertIn("is_consistent", trend)
    
    def test_prediction_accuracy(self):
        """测试预测准确率统计"""
        # 记录一些预测结果
        self.predictor.record_prediction_result("zone_2", "zone_2")  # 正确
        self.predictor.record_prediction_result("zone_2", "zone_3")  # 错误
        self.predictor.record_prediction_result("zone_2", "zone_2")  # 正确
        
        accuracy = self.predictor.get_prediction_accuracy()
        self.assertAlmostEqual(accuracy, 2/3, places=2)
    
    def test_low_confidence_prediction(self):
        """测试低置信度预测"""
        event = MotionEvent(
            timestamp=datetime.now(),
            zone_id="zone_1",
            position=(100, 100),
            direction=0.0,
            speed=1.2,
            confidence=0.3,  # 低置信度
            track_id="person_1"
        )
        
        predictions = self.predictor.predict_destination(event)
        # 低置信度应该返回空列表
        self.assertEqual(len(predictions), 0)


class TestMotionEvent(unittest.TestCase):
    """测试运动事件"""
    
    def test_motion_event_creation(self):
        """测试创建运动事件"""
        event = MotionEvent(
            timestamp=datetime.now(),
            zone_id="zone_1",
            position=(100, 100),
            direction=45.0,
            speed=1.5,
            confidence=0.85,
            track_id="person_1"
        )
        
        self.assertEqual(event.zone_id, "zone_1")
        self.assertEqual(event.direction, 45.0)
        self.assertEqual(event.speed, 1.5)
        self.assertEqual(event.confidence, 0.85)


if __name__ == '__main__':
    unittest.main()
