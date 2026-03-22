"""
查询接口模块 - 提供给交警系统的查询API
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from .congestion_llm import CongestionEvent, CongestionAnalyzer
from .rag_writer import CongestionRAGWriter

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """查询结果"""
    query_id: str
    query_text: str
    filters: Dict[str, Any]
    results: List[CongestionEvent]
    total_count: int
    execution_time_ms: float
    timestamp: datetime


class CongestionQueryInterface:
    """
    拥堵查询接口
    
    提供给交警系统的自然语言查询接口
    支持按时间、地点、原因等多维度查询
    """
    
    def __init__(
        self,
        analyzer: Optional[CongestionAnalyzer] = None,
        rag_writer: Optional[CongestionRAGWriter] = None
    ):
        """
        初始化查询接口
        
        Args:
            analyzer: 拥堵分析器
            rag_writer: RAG写入器
        """
        self.analyzer = analyzer
        self.rag_writer = rag_writer or (analyzer.rag_writer if analyzer else None)
        
        if self.rag_writer is None:
            raise ValueError("必须提供rag_writer或analyzer")
        
        logger.info("CongestionQueryInterface初始化完成")
    
    def query(
        self,
        query_text: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> QueryResult:
        """
        自然语言查询拥堵事件
        
        Args:
            query_text: 自然语言查询 (如: "昨天中关村附近的事故拥堵")
            filters: 额外过滤条件
                - time_range: (start, end) 时间范围
                - location: (lat, lon, radius) 位置和半径
                - causes: List[str] 拥堵原因列表
                - severity: List[str] 严重度列表
                - status: str 事件状态
        
        Returns:
            QueryResult: 查询结果
        """
        import time
        import uuid
        
        start_time = time.time()
        filters = filters or {}
        
        logger.info(f"执行查询: {query_text}")
        
        # 解析查询中的过滤条件
        parsed_filters = self._parse_query(query_text)
        parsed_filters.update(filters)
        
        # 查询RAG
        rag_results = self.rag_writer.query_events(query_text, parsed_filters)
        
        # 转换为CongestionEvent对象
        events = []
        for result in rag_results:
            try:
                event = self._parse_rag_result(result)
                if event and self._apply_filters(event, parsed_filters):
                    events.append(event)
            except Exception as e:
                logger.error(f"解析RAG结果失败: {e}")
                continue
        
        execution_time = (time.time() - start_time) * 1000
        
        logger.info(f"查询完成: 找到 {len(events)} 个事件，耗时 {execution_time:.2f}ms")
        
        return QueryResult(
            query_id=str(uuid.uuid4()),
            query_text=query_text,
            filters=parsed_filters,
            results=events,
            total_count=len(events),
            execution_time_ms=execution_time,
            timestamp=datetime.now()
        )
    
    def _parse_query(self, query_text: str) -> Dict[str, Any]:
        """
        解析自然语言查询中的过滤条件
        
        Args:
            query_text: 查询文本
            
        Returns:
            解析出的过滤条件
        """
        filters = {}
        query_lower = query_text.lower()
        
        # 解析时间
        now = datetime.now()
        
        if "今天" in query_text or "今日" in query_text:
            filters["time_range"] = (
                now.replace(hour=0, minute=0, second=0, microsecond=0),
                now
            )
        elif "昨天" in query_text or "昨日" in query_text:
            yesterday = now - timedelta(days=1)
            filters["time_range"] = (
                yesterday.replace(hour=0, minute=0, second=0, microsecond=0),
                yesterday.replace(hour=23, minute=59, second=59)
            )
        elif "最近" in query_text or "近" in query_text:
            # 最近24小时
            filters["time_range"] = (now - timedelta(hours=24), now)
        elif "本周" in query_text:
            week_start = now - timedelta(days=now.weekday())
            filters["time_range"] = (
                week_start.replace(hour=0, minute=0, second=0, microsecond=0),
                now
            )
        
        # 解析原因
        cause_keywords = {
            "事故": "accident",
            "撞车": "accident",
            "碰撞": "accident",
            "施工": "construction",
            "修路": "construction",
            "交警": "police_control",
            "管制": "police_control",
            "天气": "weather",
            "下雨": "weather",
            "大雾": "weather",
            "大车": "large_vehicle",
            "货车": "large_vehicle",
            "信号灯": "signal_failure",
            "红绿灯": "signal_failure"
        }
        
        for keyword, cause in cause_keywords.items():
            if keyword in query_text:
                filters["cause"] = cause
                break
        
        # 解析严重度
        severity_keywords = {
            "轻微": "low",
            "一般": "medium",
            "严重": "high",
            "紧急": "critical",
            "危险": "critical"
        }
        
        for keyword, severity in severity_keywords.items():
            if keyword in query_text:
                filters["severity"] = severity
                break
        
        return filters
    
    def _parse_rag_result(self, result: Dict[str, Any]) -> Optional[CongestionEvent]:
        """
        解析RAG查询结果为CongestionEvent
        
        Args:
            result: RAG返回的字典
            
        Returns:
            CongestionEvent对象或None
        """
        try:
            properties = result.get("properties", result)
            
            # 提取位置
            location = (0.0, 0.0)
            if "lat" in properties and "lon" in properties:
                location = (properties["lat"], properties["lon"])
            elif "location" in properties:
                loc = properties["location"]
                if isinstance(loc, (list, tuple)) and len(loc) >= 2:
                    location = (loc[0], loc[1])
            
            return CongestionEvent(
                event_id=properties.get("event_id", "unknown"),
                timestamp=datetime.fromisoformat(properties["timestamp"]) if "timestamp" in properties else datetime.now(),
                location=location,
                camera_ids=properties.get("camera_ids", []),
                density=properties.get("density", 0.0),
                avg_speed=properties.get("avg_speed", 0.0),
                vehicle_types=properties.get("vehicle_types", {}),
                duration=properties.get("duration", 0),
                cause=properties.get("cause", "unknown"),
                cause_confidence=properties.get("confidence", 0.0),
                cause_description=properties.get("description", ""),
                severity=properties.get("severity", "medium"),
                recommended_action=properties.get("recommended_action", ""),
                status=properties.get("status", "active"),
                created_at=datetime.fromisoformat(properties["created_at"]) if "created_at" in properties else datetime.now(),
                resolved_at=datetime.fromisoformat(properties["resolved_at"]) if "resolved_at" in properties else None
            )
        except Exception as e:
            logger.error(f"解析RAG结果失败: {e}")
            return None
    
    def _apply_filters(
        self,
        event: CongestionEvent,
        filters: Dict[str, Any]
    ) -> bool:
        """
        应用过滤条件
        
        Args:
            event: 拥堵事件
            filters: 过滤条件
            
        Returns:
            是否通过过滤
        """
        # 时间范围过滤
        if "time_range" in filters:
            start, end = filters["time_range"]
            if not (start <= event.timestamp <= end):
                return False
        
        # 原因过滤
        if "cause" in filters:
            if event.cause != filters["cause"]:
                return False
        
        # 严重度过滤
        if "severity" in filters:
            if event.severity != filters["severity"]:
                return False
        
        # 状态过滤
        if "status" in filters:
            if event.status != filters["status"]:
                return False
        
        # 位置过滤
        if "location" in filters:
            lat, lon, radius = filters["location"]
            distance = self._calculate_distance(
                event.location[0], event.location[1],
                lat, lon
            )
            if distance > radius:
                return False
        
        # 原因列表过滤
        if "causes" in filters:
            if event.cause not in filters["causes"]:
                return False
        
        # 严重度列表过滤
        if "severities" in filters:
            if event.severity not in filters["severities"]:
                return False
        
        return True
    
    def _calculate_distance(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """
        计算两点间距离（简化版，使用欧氏距离）
        
        Args:
            lat1, lon1: 点1坐标
            lat2, lon2: 点2坐标
            
        Returns:
            距离（米）
        """
        import math
        
        # 简化的距离计算（适用于小范围）
        # 1度纬度约111km，1度经度约111km * cos(纬度)
        lat_diff = (lat2 - lat1) * 111000
        lon_diff = (lon2 - lon1) * 111000 * math.cos(math.radians(lat1))
        
        return math.sqrt(lat_diff ** 2 + lon_diff ** 2)
    
    def get_event_detail(self, event_id: str) -> Optional[Dict[str, Any]]:
        """
        获取拥堵事件详情
        
        Args:
            event_id: 事件ID
            
        Returns:
            事件详情字典
        """
        detail = self.rag_writer.get_event_detail(event_id)
        
        if detail:
            return detail
        
        # 尝试从活跃事件缓存获取
        if self.analyzer and event_id in self.analyzer._active_events:
            event = self.analyzer._active_events[event_id]
            return {
                "event": event.to_dict(),
                "related": []
            }
        
        return None
    
    def get_active_events(
        self,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[CongestionEvent]:
        """
        获取活跃事件列表
        
        Args:
            filters: 过滤条件
            
        Returns:
            活跃事件列表
        """
        if not self.analyzer:
            return []
        
        events = self.analyzer.get_active_events()
        
        if filters:
            events = [e for e in events if self._apply_filters(e, filters)]
        
        return events
    
    def get_statistics(
        self,
        time_range: Optional[tuple] = None
    ) -> Dict[str, Any]:
        """
        获取统计信息
        
        Args:
            time_range: 时间范围 (start, end)
            
        Returns:
            统计信息字典
        """
        stats = {
            "timestamp": datetime.now().isoformat(),
            "rag_statistics": self.rag_writer.get_statistics()
        }
        
        if self.analyzer:
            stats["event_statistics"] = self.analyzer.get_event_statistics()
        
        return stats
    
    def resolve_event(self, event_id: str) -> bool:
        """
        标记事件为已解决
        
        Args:
            event_id: 事件ID
            
        Returns:
            是否成功
        """
        if self.analyzer:
            return self.analyzer.resolve_event(event_id)
        return False


# 便捷的API函数
def query_congestion(
    query_text: str,
    filters: Optional[Dict[str, Any]] = None,
    rag_writer: Optional[CongestionRAGWriter] = None
) -> QueryResult:
    """
    便捷的拥堵查询函数
    
    Args:
        query_text: 查询文本
        filters: 过滤条件
        rag_writer: RAG写入器（如未提供则创建新的）
        
    Returns:
        QueryResult: 查询结果
    """
    if rag_writer is None:
        rag_writer = CongestionRAGWriter()
    
    interface = CongestionQueryInterface(rag_writer=rag_writer)
    return interface.query(query_text, filters)


def get_event_detail(
    event_id: str,
    rag_writer: Optional[CongestionRAGWriter] = None
) -> Optional[Dict[str, Any]]:
    """
    获取事件详情
    
    Args:
        event_id: 事件ID
        rag_writer: RAG写入器
        
    Returns:
        事件详情
    """
    if rag_writer is None:
        rag_writer = CongestionRAGWriter()
    
    interface = CongestionQueryInterface(rag_writer=rag_writer)
    return interface.get_event_detail(event_id)
