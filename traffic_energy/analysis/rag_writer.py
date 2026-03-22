"""
RAG写入器模块 - 将拥堵事件写入知识图谱

支持YouTu GraphRAG和通用RAG接口
"""

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """知识图谱实体"""
    name: str
    entity_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "name": self.name,
            "type": self.entity_type,
            "properties": self.properties
        }


@dataclass
class Relation:
    """知识图谱关系"""
    source: str
    target: str
    relation_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "source": self.source,
            "target": self.target,
            "type": self.relation_type,
            "properties": self.properties
        }


class BaseGraphRAGClient:
    """基础GraphRAG客户端接口"""
    
    def add_entity(self, entity: Entity) -> bool:
        """添加实体"""
        raise NotImplementedError
    
    def add_relation(self, relation: Relation) -> bool:
        """添加关系"""
        raise NotImplementedError
    
    def query(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        """查询"""
        raise NotImplementedError


class InMemoryGraphRAG(BaseGraphRAGClient):
    """
    内存中的GraphRAG实现
    
    用于测试和降级方案
    """
    
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
    
    def add_entity(self, entity: Entity) -> bool:
        """添加实体"""
        self.entities[entity.name] = entity
        logger.debug(f"添加实体: {entity.name} ({entity.entity_type})")
        return True
    
    def add_relation(self, relation: Relation) -> bool:
        """添加关系"""
        self.relations.append(relation)
        logger.debug(f"添加关系: {relation.source} -> {relation.target} ({relation.relation_type})")
        return True
    
    def query(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        """
        简单查询实现
        
        支持关键词匹配和过滤条件
        """
        results = []
        query_lower = query.lower()
        
        for entity in self.entities.values():
            # 检查实体是否匹配查询
            match = False
            
            # 检查名称
            if query_lower in entity.name.lower():
                match = True
            
            # 检查类型
            if query_lower in entity.entity_type.lower():
                match = True
            
            # 检查属性
            for key, value in entity.properties.items():
                if query_lower in str(value).lower():
                    match = True
                    break
            
            # 应用过滤器
            if filters and match:
                for filter_key, filter_value in filters.items():
                    if filter_key in entity.properties:
                        if entity.properties[filter_key] != filter_value:
                            match = False
                            break
            
            if match:
                results.append(entity.to_dict())
        
        return results
    
    def get_related_entities(self, entity_name: str) -> List[Dict]:
        """获取相关实体"""
        related = []
        for relation in self.relations:
            if relation.source == entity_name:
                if relation.target in self.entities:
                    related.append({
                        "relation": relation.to_dict(),
                        "entity": self.entities[relation.target].to_dict()
                    })
            elif relation.target == entity_name:
                if relation.source in self.entities:
                    related.append({
                        "relation": relation.to_dict(),
                        "entity": self.entities[relation.source].to_dict()
                    })
        return related


class YouTuGraphRAGClient(BaseGraphRAGClient):
    """
    YouTu GraphRAG客户端
    
    注意: 此为占位实现，需要根据实际SDK调整
    """
    
    def __init__(self, api_key: Optional[str] = None, endpoint: Optional[str] = None):
        """
        初始化YouTu GraphRAG客户端
        
        Args:
            api_key: API密钥
            endpoint: API端点
        """
        self.api_key = api_key
        self.endpoint = endpoint or "https://api.youtu.com/graphrag"
        self._client = None
        self._init_client()
    
    def _init_client(self):
        """初始化客户端"""
        # TODO: 根据实际SDK初始化
        # 这里使用requests作为示例
        try:
            import requests
            self._client = requests
        except ImportError:
            logger.warning("requests库未安装，YouTu GraphRAG功能不可用")
            self._client = None
    
    def add_entity(self, entity: Entity) -> bool:
        """添加实体到YouTu GraphRAG"""
        if self._client is None:
            logger.error("YouTu GraphRAG客户端未初始化")
            return False
        
        try:
            # TODO: 根据实际API调整
            response = self._client.post(
                f"{self.endpoint}/entities",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=entity.to_dict()
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"添加实体失败: {e}")
            return False
    
    def add_relation(self, relation: Relation) -> bool:
        """添加关系到YouTu GraphRAG"""
        if self._client is None:
            logger.error("YouTu GraphRAG客户端未初始化")
            return False
        
        try:
            # TODO: 根据实际API调整
            response = self._client.post(
                f"{self.endpoint}/relations",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=relation.to_dict()
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"添加关系失败: {e}")
            return False
    
    def query(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        """查询YouTu GraphRAG"""
        if self._client is None:
            logger.error("YouTu GraphRAG客户端未初始化")
            return []
        
        try:
            # TODO: 根据实际API调整
            payload = {"query": query}
            if filters:
                payload["filters"] = filters
            
            response = self._client.post(
                f"{self.endpoint}/query",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload
            )
            
            if response.status_code == 200:
                return response.json().get("results", [])
            else:
                logger.error(f"查询失败: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"查询失败: {e}")
            return []


class CongestionRAGWriter:
    """
    拥堵信息RAG写入器
    
    负责将拥堵事件写入知识图谱，支持多种RAG后端
    """
    
    def __init__(
        self,
        graph_rag_client: Optional[BaseGraphRAGClient] = None,
        use_memory_fallback: bool = True
    ):
        """
        初始化RAG写入器
        
        Args:
            graph_rag_client: GraphRAG客户端，如未提供则使用内存实现
            use_memory_fallback: 当外部客户端失败时是否使用内存实现
        """
        self.rag = graph_rag_client or InMemoryGraphRAG()
        self.use_memory_fallback = use_memory_fallback
        self._memory_rag = InMemoryGraphRAG() if use_memory_fallback else None
        
        logger.info(f"CongestionRAGWriter初始化完成，后端: {type(self.rag).__name__}")
    
    def write_event(self, event: 'CongestionEvent') -> bool:
        """
        将拥堵事件写入RAG
        
        实体:
        - CongestionEvent (拥堵事件)
        - Location (位置)
        - Camera (摄像头)
        - VehicleType (车辆类型)
        
        关系:
        - occurs_at (事件发生位置)
        - detected_by (被摄像头检测)
        - involves (涉及车辆类型)
        - has_cause (原因)
        
        Args:
            event: 拥堵事件对象
            
        Returns:
            写入是否成功
        """
        try:
            # 创建拥堵事件实体
            event_entity = Entity(
                name=f"congestion_{event.event_id}",
                entity_type="CongestionEvent",
                properties={
                    "event_id": event.event_id,
                    "timestamp": event.timestamp.isoformat() if isinstance(event.timestamp, datetime) else event.timestamp,
                    "severity": event.severity,
                    "cause": event.cause,
                    "confidence": event.cause_confidence,
                    "description": event.cause_description,
                    "status": event.status,
                    "avg_speed": event.avg_speed,
                    "density": event.density,
                    "duration": event.duration
                }
            )
            
            # 创建位置实体
            location_name = f"loc_{event.location[0]}_{event.location[1]}"
            location_entity = Entity(
                name=location_name,
                entity_type="Location",
                properties={
                    "lat": event.location[0],
                    "lon": event.location[1],
                    "coordinates": f"{event.location[0]},{event.location[1]}"
                }
            )
            
            # 创建原因实体
            cause_entity = Entity(
                name=f"cause_{event.cause}_{event.event_id}",
                entity_type="CongestionCause",
                properties={
                    "type": event.cause,
                    "confidence": event.cause_confidence,
                    "description": event.cause_description
                }
            )
            
            # 创建关系
            relations = [
                Relation(
                    source=event_entity.name,
                    target=location_entity.name,
                    relation_type="occurs_at"
                ),
                Relation(
                    source=event_entity.name,
                    target=cause_entity.name,
                    relation_type="has_cause"
                )
            ]
            
            # 添加摄像头实体和关系
            for camera_id in event.camera_ids:
                camera_entity = Entity(
                    name=f"camera_{camera_id}",
                    entity_type="Camera",
                    properties={"camera_id": camera_id}
                )
                self._add_entity_safe(camera_entity)
                
                relations.append(Relation(
                    source=event_entity.name,
                    target=camera_entity.name,
                    relation_type="detected_by"
                ))
            
            # 添加车辆类型实体和关系
            for vehicle_type, count in event.vehicle_types.items():
                vehicle_entity = Entity(
                    name=f"vehicle_{vehicle_type}_{event.event_id}",
                    entity_type="VehicleType",
                    properties={
                        "type": vehicle_type,
                        "count": count
                    }
                )
                self._add_entity_safe(vehicle_entity)
                
                relations.append(Relation(
                    source=event_entity.name,
                    target=vehicle_entity.name,
                    relation_type="involves"
                ))
            
            # 写入所有实体和关系
            self._add_entity_safe(event_entity)
            self._add_entity_safe(location_entity)
            self._add_entity_safe(cause_entity)
            
            for relation in relations:
                self._add_relation_safe(relation)
            
            logger.info(f"拥堵事件写入成功: {event.event_id}")
            return True
            
        except Exception as e:
            logger.error(f"写入拥堵事件失败: {e}")
            return False
    
    def _add_entity_safe(self, entity: Entity) -> bool:
        """安全地添加实体"""
        try:
            result = self.rag.add_entity(entity)
            if not result and self._memory_rag:
                logger.warning(f"主RAG写入失败，使用内存RAG: {entity.name}")
                return self._memory_rag.add_entity(entity)
            return result
        except Exception as e:
            logger.error(f"添加实体失败: {e}")
            if self._memory_rag:
                return self._memory_rag.add_entity(entity)
            return False
    
    def _add_relation_safe(self, relation: Relation) -> bool:
        """安全地添加关系"""
        try:
            result = self.rag.add_relation(relation)
            if not result and self._memory_rag:
                logger.warning(f"主RAG关系写入失败，使用内存RAG: {relation.relation_type}")
                return self._memory_rag.add_relation(relation)
            return result
        except Exception as e:
            logger.error(f"添加关系失败: {e}")
            if self._memory_rag:
                return self._memory_rag.add_relation(relation)
            return False
    
    def query_events(
        self,
        query: str,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        查询拥堵事件
        
        Args:
            query: 自然语言查询 (如: "昨天中关村附近的事故拥堵")
            filters: 过滤条件
            
        Returns:
            拥堵事件列表
        """
        try:
            # 使用GraphRAG查询
            results = self.rag.query(query, filters)
            
            # 如果主RAG没有结果且内存RAG有数据，尝试内存RAG
            if not results and self._memory_rag:
                results = self._memory_rag.query(query, filters)
            
            return results
            
        except Exception as e:
            logger.error(f"查询失败: {e}")
            # 尝试内存RAG
            if self._memory_rag:
                return self._memory_rag.query(query, filters)
            return []
    
    def get_event_detail(self, event_id: str) -> Optional[Dict]:
        """
        获取拥堵事件详情
        
        Args:
            event_id: 事件ID
            
        Returns:
            事件详情字典
        """
        entity_name = f"congestion_{event_id}"
        
        # 尝试从主RAG获取
        if isinstance(self.rag, InMemoryGraphRAG):
            if entity_name in self.rag.entities:
                entity = self.rag.entities[entity_name]
                related = self.rag.get_related_entities(entity_name)
                return {
                    "event": entity.to_dict(),
                    "related": related
                }
        
        # 尝试从内存RAG获取
        if self._memory_rag and entity_name in self._memory_rag.entities:
            entity = self._memory_rag.entities[entity_name]
            related = self._memory_rag.get_related_entities(entity_name)
            return {
                "event": entity.to_dict(),
                "related": related
            }
        
        return None
    
    def get_statistics(self) -> Dict:
        """
        获取RAG统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            "backend_type": type(self.rag).__name__,
            "memory_fallback_enabled": self._memory_rag is not None
        }
        
        if isinstance(self.rag, InMemoryGraphRAG):
            stats["entity_count"] = len(self.rag.entities)
            stats["relation_count"] = len(self.rag.relations)
        
        if self._memory_rag:
            stats["memory_entity_count"] = len(self._memory_rag.entities)
            stats["memory_relation_count"] = len(self._memory_rag.relations)
        
        return stats
