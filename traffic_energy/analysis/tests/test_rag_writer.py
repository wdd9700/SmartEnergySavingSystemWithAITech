"""
RAG写入器测试
"""

import unittest
from datetime import datetime
from unittest.mock import Mock, patch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis.rag_writer import (
    Entity,
    Relation,
    InMemoryGraphRAG,
    CongestionRAGWriter,
    YouTuGraphRAGClient
)
from analysis.congestion_llm import CongestionEvent


class TestEntity(unittest.TestCase):
    """测试Entity数据类"""
    
    def test_creation(self):
        """测试创建实体"""
        entity = Entity(
            name="test_entity",
            entity_type="TestType",
            properties={"key": "value"}
        )
        self.assertEqual(entity.name, "test_entity")
        self.assertEqual(entity.entity_type, "TestType")
        self.assertEqual(entity.properties["key"], "value")
    
    def test_to_dict(self):
        """测试转换为字典"""
        entity = Entity(
            name="test",
            entity_type="Type",
            properties={"a": 1}
        )
        result = entity.to_dict()
        self.assertEqual(result["name"], "test")
        self.assertEqual(result["type"], "Type")
        self.assertEqual(result["properties"]["a"], 1)


class TestRelation(unittest.TestCase):
    """测试Relation数据类"""
    
    def test_creation(self):
        """测试创建关系"""
        relation = Relation(
            source="entity1",
            target="entity2",
            relation_type="connected_to",
            properties={"weight": 1.0}
        )
        self.assertEqual(relation.source, "entity1")
        self.assertEqual(relation.target, "entity2")
        self.assertEqual(relation.relation_type, "connected_to")
    
    def test_to_dict(self):
        """测试转换为字典"""
        relation = Relation(
            source="a",
            target="b",
            relation_type="link"
        )
        result = relation.to_dict()
        self.assertEqual(result["source"], "a")
        self.assertEqual(result["target"], "b")
        self.assertEqual(result["type"], "link")


class TestInMemoryGraphRAG(unittest.TestCase):
    """测试内存GraphRAG实现"""
    
    def setUp(self):
        """测试前准备"""
        self.rag = InMemoryGraphRAG()
    
    def test_add_entity(self):
        """测试添加实体"""
        entity = Entity(name="test", entity_type="Test")
        result = self.rag.add_entity(entity)
        self.assertTrue(result)
        self.assertIn("test", self.rag.entities)
    
    def test_add_relation(self):
        """测试添加关系"""
        relation = Relation(source="a", target="b", relation_type="link")
        result = self.rag.add_relation(relation)
        self.assertTrue(result)
        self.assertEqual(len(self.rag.relations), 1)
    
    def test_query_by_name(self):
        """测试按名称查询"""
        entity = Entity(name="congestion_123", entity_type="CongestionEvent")
        self.rag.add_entity(entity)
        
        results = self.rag.query("congestion")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "congestion_123")
    
    def test_query_by_type(self):
        """测试按类型查询"""
        entity = Entity(name="test", entity_type="Location")
        self.rag.add_entity(entity)
        
        results = self.rag.query("Location")
        self.assertEqual(len(results), 1)
    
    def test_query_by_property(self):
        """测试按属性查询"""
        entity = Entity(
            name="test",
            entity_type="Event",
            properties={"cause": "accident"}
        )
        self.rag.add_entity(entity)
        
        results = self.rag.query("accident")
        self.assertEqual(len(results), 1)
    
    def test_query_with_filters(self):
        """测试带过滤器的查询"""
        entity = Entity(
            name="test",
            entity_type="Event",
            properties={"status": "active", "severity": "high"}
        )
        self.rag.add_entity(entity)
        
        results = self.rag.query("test", filters={"status": "active"})
        self.assertEqual(len(results), 1)
        
        results = self.rag.query("test", filters={"status": "resolved"})
        self.assertEqual(len(results), 0)
    
    def test_get_related_entities(self):
        """测试获取相关实体"""
        # 添加实体
        entity1 = Entity(name="event1", entity_type="Event")
        entity2 = Entity(name="location1", entity_type="Location")
        self.rag.add_entity(entity1)
        self.rag.add_entity(entity2)
        
        # 添加关系
        relation = Relation(source="event1", target="location1", relation_type="occurs_at")
        self.rag.add_relation(relation)
        
        # 获取相关实体
        related = self.rag.get_related_entities("event1")
        self.assertEqual(len(related), 1)
        self.assertEqual(related[0]["entity"]["name"], "location1")


class TestCongestionRAGWriter(unittest.TestCase):
    """测试CongestionRAGWriter"""
    
    def setUp(self):
        """测试前准备"""
        self.rag = InMemoryGraphRAG()
        self.writer = CongestionRAGWriter(graph_rag_client=self.rag)
        
        self.test_event = CongestionEvent(
            event_id="test-123",
            timestamp=datetime.now(),
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
    
    def test_write_event(self):
        """测试写入事件"""
        result = self.writer.write_event(self.test_event)
        self.assertTrue(result)
        
        # 验证实体已添加
        self.assertIn("congestion_test-123", self.rag.entities)
        self.assertIn("loc_39.9_116.4", self.rag.entities)
        self.assertIn("cause_accident_test-123", self.rag.entities)
    
    def test_write_event_creates_relations(self):
        """测试写入事件创建关系"""
        self.writer.write_event(self.test_event)
        
        # 验证关系已添加
        self.assertEqual(len(self.rag.relations), 4)  # occurs_at + has_cause + 2 cameras
    
    def test_query_events(self):
        """测试查询事件"""
        self.writer.write_event(self.test_event)
        
        results = self.writer.query_events("accident")
        self.assertGreater(len(results), 0)
    
    def test_get_event_detail(self):
        """测试获取事件详情"""
        self.writer.write_event(self.test_event)
        
        detail = self.writer.get_event_detail("test-123")
        self.assertIsNotNone(detail)
        self.assertEqual(detail["event"]["name"], "congestion_test-123")
    
    def test_get_event_detail_not_found(self):
        """测试获取不存在的事件详情"""
        detail = self.writer.get_event_detail("nonexistent")
        self.assertIsNone(detail)
    
    def test_get_statistics(self):
        """测试获取统计信息"""
        self.writer.write_event(self.test_event)
        
        stats = self.writer.get_statistics()
        self.assertIn("entity_count", stats)
        self.assertIn("relation_count", stats)
        self.assertEqual(stats["entity_count"], 6)  # event + location + cause + 2 cameras + 2 vehicle types
    
    def test_memory_fallback(self):
        """测试内存降级方案"""
        # 创建一个会失败的RAG客户端
        failing_rag = Mock()
        failing_rag.add_entity.return_value = False
        failing_rag.add_relation.return_value = False
        
        writer = CongestionRAGWriter(
            graph_rag_client=failing_rag,
            use_memory_fallback=True
        )
        
        result = writer.write_event(self.test_event)
        # 即使主RAG失败，也应该通过内存RAG成功
        self.assertTrue(result)


class TestYouTuGraphRAGClient(unittest.TestCase):
    """测试YouTu GraphRAG客户端"""
    
    @patch('analysis.rag_writer.YouTuGraphRAGClient._init_client')
    def test_initialization(self, mock_init):
        """测试初始化"""
        mock_init.return_value = None
        client = YouTuGraphRAGClient(api_key="test_key", endpoint="https://test.com")
        self.assertEqual(client.api_key, "test_key")
        self.assertEqual(client.endpoint, "https://test.com")
    
    @patch('analysis.rag_writer.YouTuGraphRAGClient._init_client')
    def test_add_entity_not_initialized(self, mock_init):
        """测试未初始化时添加实体"""
        mock_init.return_value = None
        client = YouTuGraphRAGClient()
        client._client = None
        
        entity = Entity(name="test", entity_type="Test")
        result = client.add_entity(entity)
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
