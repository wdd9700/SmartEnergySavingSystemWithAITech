"""
拥堵分析器测试
"""

import unittest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from analysis.congestion_llm import (
    CongestionEvent,
    CongestionHotspot,
    CongestionAnalyzer
)
from analysis.vlm_client import CongestionAnalysisResult


class TestCongestionEvent(unittest.TestCase):
    """测试CongestionEvent数据类"""
    
    def setUp(self):
        """测试前准备"""
        self.now = datetime.now()
        self.event = CongestionEvent(
            event_id="test-123",
            timestamp=self.now,
            location=(39.9, 116.4),
            camera_ids=["cam1", "cam2"],
            density=1.5,
            avg_speed=15.0,
            vehicle_types={"car": 10, "truck": 2},
            duration=30,
            cause="accident",
            cause_confidence=0.85,
            cause_description="Vehicle collision",
            severity="high",
            recommended_action="Dispatch emergency services",
            status="active"
        )
    
    def test_creation(self):
        """测试创建事件"""
        self.assertEqual(self.event.event_id, "test-123")
        self.assertEqual(self.event.location, (39.9, 116.4))
        self.assertEqual(self.event.cause, "accident")
    
    def test_to_dict(self):
        """测试转换为字典"""
        result = self.event.to_dict()
        self.assertEqual(result["event_id"], "test-123")
        self.assertEqual(result["cause"], "accident")
        self.assertEqual(result["severity"], "high")
        self.assertIn("timestamp", result)
    
    def test_from_dict(self):
        """测试从字典创建"""
        data = self.event.to_dict()
        restored = CongestionEvent.from_dict(data)
        self.assertEqual(restored.event_id, self.event.event_id)
        self.assertEqual(restored.cause, self.event.cause)
        self.assertEqual(restored.location, self.event.location)


class TestCongestionHotspot(unittest.TestCase):
    """测试CongestionHotspot数据类"""
    
    def test_creation(self):
        """测试创建热点"""
        hotspot = CongestionHotspot(
            hotspot_id="hotspot-1",
            location=(39.9, 116.4),
            camera_ids=["cam1"],
            density=1.5,
            avg_speed=15.0,
            vehicle_types={"car": 5},
            duration=20,
            severity_score=0.7
        )
        self.assertEqual(hotspot.hotspot_id, "hotspot-1")
        self.assertEqual(hotspot.severity_score, 0.7)
    
    def test_to_traffic_data(self):
        """测试转换为交通数据"""
        hotspot = CongestionHotspot(
            hotspot_id="hotspot-1",
            location=(39.9, 116.4),
            camera_ids=["cam1"],
            density=1.5,
            avg_speed=15.0,
            vehicle_types={"car": 5},
            duration=20,
            severity_score=0.7
        )
        data = hotspot.to_traffic_data()
        self.assertEqual(data["density"], 1.5)
        self.assertEqual(data["avg_speed"], 15.0)
        self.assertEqual(data["vehicle_types"], {"car": 5})
        self.assertEqual(data["duration"], 20)


class TestCongestionAnalyzer(unittest.TestCase):
    """测试CongestionAnalyzer"""
    
    def setUp(self):
        """测试前准备"""
        self.mock_vlm = Mock()
        self.mock_rag = Mock()
        
        self.analyzer = CongestionAnalyzer(
            vlm_client=self.mock_vlm,
            rag_writer=self.mock_rag
        )
        
        self.hotspot = CongestionHotspot(
            hotspot_id="hotspot-1",
            location=(39.9, 116.4),
            camera_ids=["cam1"],
            density=1.5,
            avg_speed=15.0,
            vehicle_types={"car": 5},
            duration=20,
            severity_score=0.7
        )
        
        self.test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.analyzer.vlm_client, self.mock_vlm)
        self.assertEqual(self.analyzer.rag_writer, self.mock_rag)
        self.assertEqual(self.analyzer.confidence_threshold, 0.6)
    
    def test_calculate_severity_critical(self):
        """测试严重度计算 - 严重"""
        analysis_result = CongestionAnalysisResult(
            cause="accident",
            confidence=0.9,
            description="test",
            recommended_action="test"
        )
        
        hotspot = CongestionHotspot(
            hotspot_id="test",
            location=(0, 0),
            camera_ids=["cam1"],
            density=2.0,  # 高密度
            avg_speed=5.0,  # 低速
            vehicle_types={},
            duration=60,  # 长时间
            severity_score=0.9
        )
        
        severity = self.analyzer._calculate_severity(hotspot, analysis_result)
        self.assertIn(severity, ["high", "critical"])
    
    def test_calculate_severity_low(self):
        """测试严重度计算 - 轻微"""
        analysis_result = CongestionAnalysisResult(
            cause="unknown",
            confidence=0.3,
            description="test",
            recommended_action="test"
        )
        
        hotspot = CongestionHotspot(
            hotspot_id="test",
            location=(0, 0),
            camera_ids=["cam1"],
            density=0.3,  # 低密度
            avg_speed=50.0,  # 高速
            vehicle_types={},
            duration=5,  # 短时间
            severity_score=0.1
        )
        
        severity = self.analyzer._calculate_severity(hotspot, analysis_result)
        self.assertIn(severity, ["low", "medium"])
    
    @patch('analysis.congestion_llm.uuid.uuid4')
    def test_analyze_hotspot(self, mock_uuid):
        """测试分析热点"""
        mock_uuid.return_value = "test-uuid"
        
        # 设置VLM返回结果
        self.mock_vlm.analyze_congestion.return_value = CongestionAnalysisResult(
            cause="accident",
            confidence=0.85,
            description="Collision detected",
            recommended_action="Dispatch"
        )
        
        # 设置RAG写入成功
        self.mock_rag.write_event.return_value = True
        
        event = self.analyzer.analyze_hotspot(self.hotspot, self.test_image)
        
        self.assertEqual(event.event_id, "test-uuid")
        self.assertEqual(event.cause, "accident")
        self.assertEqual(event.cause_confidence, 0.85)
        self.assertEqual(event.status, "active")
        
        # 验证VLM被调用
        self.mock_vlm.analyze_congestion.assert_called_once()
        
        # 验证RAG被调用
        self.mock_rag.write_event.assert_called_once()
    
    def test_analyze_hotspot_batch(self):
        """测试批量分析热点"""
        hotspots = [
            CongestionHotspot(
                hotspot_id=f"hotspot-{i}",
                location=(39.9, 116.4),
                camera_ids=["cam1"],
                density=1.0,
                avg_speed=20.0,
                vehicle_types={},
                duration=10,
                severity_score=0.5
            )
            for i in range(3)
        ]
        images = [self.test_image] * 3
        
        self.mock_vlm.analyze_congestion.return_value = CongestionAnalysisResult(
            cause="construction",
            confidence=0.7,
            description="Road work",
            recommended_action="Reroute"
        )
        self.mock_rag.write_event.return_value = True
        
        events = self.analyzer.analyze_hotspot_batch(hotspots, images)
        self.assertEqual(len(events), 3)
    
    def test_analyze_hotspot_batch_mismatch(self):
        """测试批量分析数量不匹配"""
        with self.assertRaises(ValueError):
            self.analyzer.analyze_hotspot_batch([self.hotspot], [self.test_image, self.test_image])
    
    def test_resolve_event(self):
        """测试解决事件"""
        # 先添加一个活跃事件
        event = CongestionEvent(
            event_id="test-event",
            timestamp=datetime.now(),
            location=(0, 0),
            camera_ids=["cam1"],
            density=1.0,
            avg_speed=10.0,
            vehicle_types={},
            duration=30,
            cause="accident",
            cause_confidence=0.8,
            cause_description="test",
            status="active"
        )
        self.analyzer._active_events["test-event"] = event
        self.mock_rag.write_event.return_value = True
        
        result = self.analyzer.resolve_event("test-event")
        self.assertTrue(result)
        self.assertEqual(event.status, "resolved")
        self.assertIsNotNone(event.resolved_at)
    
    def test_resolve_event_not_found(self):
        """测试解决不存在的事件"""
        result = self.analyzer.resolve_event("nonexistent")
        self.assertFalse(result)
    
    def test_get_active_events(self):
        """测试获取活跃事件"""
        event = CongestionEvent(
            event_id="active-event",
            timestamp=datetime.now(),
            location=(0, 0),
            camera_ids=["cam1"],
            density=1.0,
            avg_speed=10.0,
            vehicle_types={},
            duration=30,
            cause="accident",
            cause_confidence=0.8,
            cause_description="test",
            status="active"
        )
        self.analyzer._active_events["active-event"] = event
        
        events = self.analyzer.get_active_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_id, "active-event")
    
    def test_get_event_statistics(self):
        """测试获取事件统计"""
        # 添加测试事件
        events = [
            CongestionEvent(
                event_id=f"event-{i}",
                timestamp=datetime.now(),
                location=(0, 0),
                camera_ids=["cam1"],
                density=1.0,
                avg_speed=10.0,
                vehicle_types={},
                duration=30,
                cause="accident" if i % 2 == 0 else "construction",
                cause_confidence=0.8,
                cause_description="test",
                severity="high" if i % 2 == 0 else "medium",
                status="active"
            )
            for i in range(4)
        ]
        
        for event in events:
            self.analyzer._active_events[event.event_id] = event
        
        self.mock_rag.get_statistics.return_value = {"entity_count": 10}
        
        stats = self.analyzer.get_event_statistics()
        self.assertEqual(stats["total_active"], 4)
        self.assertIn("accident", stats["by_cause"])
        self.assertIn("construction", stats["by_cause"])
        self.assertIn("high", stats["by_severity"])
        self.assertIn("medium", stats["by_severity"])
    
    def test_health_check(self):
        """测试健康检查"""
        self.mock_vlm.health_check.return_value = {"status": "ready"}
        self.mock_rag.get_statistics.return_value = {"entity_count": 5}
        
        health = self.analyzer.health_check()
        self.assertIn("vlm_client", health)
        self.assertIn("rag_writer", health)
        self.assertIn("active_events_count", health)


if __name__ == '__main__':
    unittest.main()
