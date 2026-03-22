"""
查询接口测试
"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis.query_interface import (
    QueryResult,
    CongestionQueryInterface,
    query_congestion,
    get_event_detail
)
from analysis.congestion_llm import CongestionEvent


class TestQueryResult(unittest.TestCase):
    """测试QueryResult数据类"""
    
    def test_creation(self):
        """测试创建查询结果"""
        now = datetime.now()
        result = QueryResult(
            query_id="query-123",
            query_text="test query",
            filters={},
            results=[],
            total_count=0,
            execution_time_ms=100.0,
            timestamp=now
        )
        self.assertEqual(result.query_id, "query-123")
        self.assertEqual(result.total_count, 0)
        self.assertEqual(result.execution_time_ms, 100.0)


class TestCongestionQueryInterface(unittest.TestCase):
    """测试CongestionQueryInterface"""
    
    def setUp(self):
        """测试前准备"""
        self.mock_analyzer = Mock()
        self.mock_rag = Mock()
        self.mock_analyzer.rag_writer = self.mock_rag
        
        self.interface = CongestionQueryInterface(
            analyzer=self.mock_analyzer,
            rag_writer=self.mock_rag
        )
    
    def test_initialization_without_analyzer(self):
        """测试不使用analyzer初始化"""
        interface = CongestionQueryInterface(rag_writer=self.mock_rag)
        self.assertIsNone(interface.analyzer)
        self.assertEqual(interface.rag_writer, self.mock_rag)
    
    def test_initialization_without_rag_writer(self):
        """测试不提供rag_writer时应该抛出异常"""
        with self.assertRaises(ValueError):
            CongestionQueryInterface()
    
    def test_parse_query_time_today(self):
        """测试解析今天的时间"""
        filters = self.interface._parse_query("今天的事故拥堵")
        self.assertIn("time_range", filters)
        start, end = filters["time_range"]
        self.assertIsInstance(start, datetime)
        self.assertIsInstance(end, datetime)
    
    def test_parse_query_time_yesterday(self):
        """测试解析昨天的时间"""
        filters = self.interface._parse_query("昨天的拥堵")
        self.assertIn("time_range", filters)
        start, end = filters["time_range"]
        # 验证是昨天的时间
        now = datetime.now()
        yesterday = now - timedelta(days=1)
        self.assertEqual(start.day, yesterday.day)
    
    def test_parse_query_cause_accident(self):
        """测试解析事故原因"""
        filters = self.interface._parse_query("中关村附近的事故")
        self.assertEqual(filters.get("cause"), "accident")
    
    def test_parse_query_cause_construction(self):
        """测试解析施工原因"""
        filters = self.interface._parse_query("施工导致的拥堵")
        self.assertEqual(filters.get("cause"), "construction")
    
    def test_parse_query_cause_weather(self):
        """测试解析天气原因"""
        filters = self.interface._parse_query("下雨造成的拥堵")
        self.assertEqual(filters.get("cause"), "weather")
    
    def test_parse_query_severity(self):
        """测试解析严重度"""
        filters = self.interface._parse_query("严重的拥堵")
        self.assertEqual(filters.get("severity"), "high")
    
    def test_parse_query_no_filters(self):
        """测试无过滤条件的查询"""
        filters = self.interface._parse_query("所有拥堵事件")
        self.assertEqual(filters, {})
    
    def test_apply_filters_time_range(self):
        """测试时间范围过滤"""
        now = datetime.now()
        event = CongestionEvent(
            event_id="test",
            timestamp=now,
            location=(0, 0),
            camera_ids=[],
            density=1.0,
            avg_speed=10.0,
            vehicle_types={},
            duration=30,
            cause="accident",
            cause_confidence=0.8,
            cause_description="test"
        )
        
        # 应该通过
        filters = {"time_range": (now - timedelta(hours=1), now + timedelta(hours=1))}
        self.assertTrue(self.interface._apply_filters(event, filters))
        
        # 不应该通过
        filters = {"time_range": (now + timedelta(hours=1), now + timedelta(hours=2))}
        self.assertFalse(self.interface._apply_filters(event, filters))
    
    def test_apply_filters_cause(self):
        """测试原因过滤"""
        event = CongestionEvent(
            event_id="test",
            timestamp=datetime.now(),
            location=(0, 0),
            camera_ids=[],
            density=1.0,
            avg_speed=10.0,
            vehicle_types={},
            duration=30,
            cause="accident",
            cause_confidence=0.8,
            cause_description="test"
        )
        
        self.assertTrue(self.interface._apply_filters(event, {"cause": "accident"}))
        self.assertFalse(self.interface._apply_filters(event, {"cause": "weather"}))
    
    def test_apply_filters_severity(self):
        """测试严重度过滤"""
        event = CongestionEvent(
            event_id="test",
            timestamp=datetime.now(),
            location=(0, 0),
            camera_ids=[],
            density=1.0,
            avg_speed=10.0,
            vehicle_types={},
            duration=30,
            cause="accident",
            cause_confidence=0.8,
            cause_description="test",
            severity="high"
        )
        
        self.assertTrue(self.interface._apply_filters(event, {"severity": "high"}))
        self.assertFalse(self.interface._apply_filters(event, {"severity": "low"}))
    
    def test_apply_filters_status(self):
        """测试状态过滤"""
        event = CongestionEvent(
            event_id="test",
            timestamp=datetime.now(),
            location=(0, 0),
            camera_ids=[],
            density=1.0,
            avg_speed=10.0,
            vehicle_types={},
            duration=30,
            cause="accident",
            cause_confidence=0.8,
            cause_description="test",
            status="active"
        )
        
        self.assertTrue(self.interface._apply_filters(event, {"status": "active"}))
        self.assertFalse(self.interface._apply_filters(event, {"status": "resolved"}))
    
    def test_apply_filters_causes_list(self):
        """测试原因列表过滤"""
        event = CongestionEvent(
            event_id="test",
            timestamp=datetime.now(),
            location=(0, 0),
            camera_ids=[],
            density=1.0,
            avg_speed=10.0,
            vehicle_types={},
            duration=30,
            cause="accident",
            cause_confidence=0.8,
            cause_description="test"
        )
        
        self.assertTrue(self.interface._apply_filters(event, {"causes": ["accident", "weather"]}))
        self.assertFalse(self.interface._apply_filters(event, {"causes": ["weather", "construction"]}))
    
    def test_calculate_distance(self):
        """测试距离计算"""
        # 北京天安门附近两点
        distance = self.interface._calculate_distance(
            39.9042, 116.4074,
            39.9142, 116.4174
        )
        # 大约1.4km
        self.assertGreater(distance, 1000)
        self.assertLess(distance, 2000)
    
    def test_parse_rag_result(self):
        """测试解析RAG结果"""
        rag_result = {
            "name": "congestion_test",
            "properties": {
                "event_id": "test-123",
                "timestamp": datetime.now().isoformat(),
                "cause": "accident",
                "confidence": 0.85,
                "severity": "high",
                "status": "active",
                "lat": 39.9,
                "lon": 116.4
            }
        }
        
        event = self.interface._parse_rag_result(rag_result)
        self.assertIsNotNone(event)
        self.assertEqual(event.event_id, "test-123")
        self.assertEqual(event.cause, "accident")
    
    def test_parse_rag_result_invalid(self):
        """测试解析无效的RAG结果"""
        rag_result = {"invalid": "data"}
        event = self.interface._parse_rag_result(rag_result)
        # 应该返回一个事件，但使用默认值
        self.assertIsNotNone(event)
    
    def test_query(self):
        """测试查询"""
        now = datetime.now()
        mock_event_data = {
            "name": "congestion_test",
            "properties": {
                "event_id": "test-123",
                "timestamp": now.isoformat(),
                "cause": "accident",
                "confidence": 0.85,
                "severity": "high",
                "status": "active",
                "lat": 39.9,
                "lon": 116.4,
                "camera_ids": ["cam1"],
                "density": 1.5,
                "avg_speed": 15.0,
                "vehicle_types": {},
                "duration": 30
            }
        }
        
        self.mock_rag.query_events.return_value = [mock_event_data]
        
        result = self.interface.query("今天的事故拥堵")
        
        self.assertIsInstance(result, QueryResult)
        self.assertEqual(result.query_text, "今天的事故拥堵")
        self.assertEqual(len(result.results), 1)
        self.assertEqual(result.results[0].event_id, "test-123")
    
    def test_get_event_detail(self):
        """测试获取事件详情"""
        mock_detail = {
            "event": {"name": "congestion_test-123"},
            "related": []
        }
        self.mock_rag.get_event_detail.return_value = mock_detail
        
        detail = self.interface.get_event_detail("test-123")
        self.assertEqual(detail["event"]["name"], "congestion_test-123")
    
    def test_get_event_detail_from_active(self):
        """测试从活跃事件获取详情"""
        self.mock_rag.get_event_detail.return_value = None
        
        event = CongestionEvent(
            event_id="active-123",
            timestamp=datetime.now(),
            location=(0, 0),
            camera_ids=[],
            density=1.0,
            avg_speed=10.0,
            vehicle_types={},
            duration=30,
            cause="accident",
            cause_confidence=0.8,
            cause_description="test"
        )
        self.mock_analyzer._active_events = {"active-123": event}
        
        detail = self.interface.get_event_detail("active-123")
        self.assertIsNotNone(detail)
        self.assertEqual(detail["event"]["event_id"], "active-123")
    
    def test_get_active_events(self):
        """测试获取活跃事件"""
        events = [
            CongestionEvent(
                event_id=f"event-{i}",
                timestamp=datetime.now(),
                location=(0, 0),
                camera_ids=[],
                density=1.0,
                avg_speed=10.0,
                vehicle_types={},
                duration=30,
                cause="accident",
                cause_confidence=0.8,
                cause_description="test",
                status="active"
            )
            for i in range(3)
        ]
        self.mock_analyzer.get_active_events.return_value = events
        
        result = self.interface.get_active_events()
        self.assertEqual(len(result), 3)
    
    def test_get_active_events_with_filters(self):
        """测试带过滤器的获取活跃事件"""
        events = [
            CongestionEvent(
                event_id="event-1",
                timestamp=datetime.now(),
                location=(0, 0),
                camera_ids=[],
                density=1.0,
                avg_speed=10.0,
                vehicle_types={},
                duration=30,
                cause="accident",
                cause_confidence=0.8,
                cause_description="test",
                severity="high",
                status="active"
            ),
            CongestionEvent(
                event_id="event-2",
                timestamp=datetime.now(),
                location=(0, 0),
                camera_ids=[],
                density=1.0,
                avg_speed=10.0,
                vehicle_types={},
                duration=30,
                cause="weather",
                cause_confidence=0.8,
                cause_description="test",
                severity="low",
                status="active"
            )
        ]
        self.mock_analyzer.get_active_events.return_value = events
        
        result = self.interface.get_active_events(filters={"cause": "accident"})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].event_id, "event-1")
    
    def test_get_statistics(self):
        """测试获取统计信息"""
        self.mock_rag.get_statistics.return_value = {"entity_count": 10}
        self.mock_analyzer.get_event_statistics.return_value = {"total_active": 5}
        
        stats = self.interface.get_statistics()
        self.assertIn("rag_statistics", stats)
        self.assertIn("event_statistics", stats)
    
    def test_resolve_event(self):
        """测试解决事件"""
        self.mock_analyzer.resolve_event.return_value = True
        
        result = self.interface.resolve_event("test-123")
        self.assertTrue(result)
        self.mock_analyzer.resolve_event.assert_called_once_with("test-123")


class TestConvenienceFunctions(unittest.TestCase):
    """测试便捷函数"""
    
    @patch('analysis.query_interface.CongestionQueryInterface')
    def test_query_congestion(self, mock_interface_class):
        """测试query_congestion函数"""
        mock_interface = Mock()
        mock_interface_class.return_value = mock_interface
        
        mock_result = Mock()
        mock_result.query_id = "test"
        mock_interface.query.return_value = mock_result
        
        result = query_congestion("今天的事故")
        mock_interface.query.assert_called_once_with("今天的事故", None)
    
    @patch('analysis.query_interface.CongestionQueryInterface')
    def test_get_event_detail(self, mock_interface_class):
        """测试get_event_detail函数"""
        mock_interface = Mock()
        mock_interface_class.return_value = mock_interface
        
        mock_detail = {"event": {"id": "test"}}
        mock_interface.get_event_detail.return_value = mock_detail
        
        result = get_event_detail("test-123")
        mock_interface.get_event_detail.assert_called_once_with("test-123")


if __name__ == '__main__':
    unittest.main()
