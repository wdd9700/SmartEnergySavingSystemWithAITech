"""
拥堵LLM分析模块 - 整合VLM分析和RAG写入的完整流程
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from .vlm_client import VLMClient, CongestionAnalysisResult
from .rag_writer import CongestionRAGWriter, BaseGraphRAGClient, InMemoryGraphRAG

logger = logging.getLogger(__name__)


@dataclass
class CongestionEvent:
    """拥堵事件数据结构"""
    event_id: str
    timestamp: datetime
    location: Tuple[float, float]  # GPS坐标
    camera_ids: List[str]         # 相关摄像头
    
    # 交通数据 (来自Module 3B)
    density: float               # 车辆密度
    avg_speed: float            # 平均速度
    vehicle_types: Dict[str, int]  # 车型分布
    duration: int               # 持续时间 (分钟)
    
    # VLM分析结果
    cause: str                  # 拥堵原因
    cause_confidence: float     # 置信度
    cause_description: str      # 原因描述
    image_url: Optional[str] = None  # 分析图像URL
    
    # 处置信息
    severity: str = "medium"    # "low" | "medium" | "high" | "critical"
    recommended_action: str = ""  # 建议措施
    status: str = "active"      # "active" | "resolved"
    
    # 时间戳
    created_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            "location": self.location,
            "camera_ids": self.camera_ids,
            "density": self.density,
            "avg_speed": self.avg_speed,
            "vehicle_types": self.vehicle_types,
            "duration": self.duration,
            "cause": self.cause,
            "cause_confidence": self.cause_confidence,
            "cause_description": self.cause_description,
            "image_url": self.image_url,
            "severity": self.severity,
            "recommended_action": self.recommended_action,
            "status": self.status,
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            "resolved_at": self.resolved_at.isoformat() if isinstance(self.resolved_at, datetime) else self.resolved_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CongestionEvent':
        """从字典创建"""
        # 解析时间戳
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now()
        
        resolved_at = data.get("resolved_at")
        if isinstance(resolved_at, str):
            resolved_at = datetime.fromisoformat(resolved_at)
        
        return cls(
            event_id=data["event_id"],
            timestamp=timestamp,
            location=tuple(data["location"]),
            camera_ids=data.get("camera_ids", []),
            density=data.get("density", 0.0),
            avg_speed=data.get("avg_speed", 0.0),
            vehicle_types=data.get("vehicle_types", {}),
            duration=data.get("duration", 0),
            cause=data.get("cause", "unknown"),
            cause_confidence=data.get("cause_confidence", 0.0),
            cause_description=data.get("cause_description", ""),
            image_url=data.get("image_url"),
            severity=data.get("severity", "medium"),
            recommended_action=data.get("recommended_action", ""),
            status=data.get("status", "active"),
            created_at=created_at,
            resolved_at=resolved_at
        )


@dataclass
class CongestionHotspot:
    """拥堵热点数据 (来自Module 3B)"""
    hotspot_id: str
    location: Tuple[float, float]
    camera_ids: List[str]
    density: float
    avg_speed: float
    vehicle_types: Dict[str, int]
    duration: int
    severity_score: float
    
    def to_traffic_data(self) -> Dict[str, Any]:
        """转换为交通数据字典"""
        return {
            "density": self.density,
            "avg_speed": self.avg_speed,
            "vehicle_types": self.vehicle_types,
            "duration": self.duration
        }


class CongestionAnalyzer:
    """
    拥堵分析器
    
    整合VLM分析和RAG写入的完整流程
    """
    
    # 严重度阈值
    SEVERITY_THRESHOLDS = {
        "low": (0, 0.3),
        "medium": (0.3, 0.6),
        "high": (0.6, 0.8),
        "critical": (0.8, 1.0)
    }
    
    def __init__(
        self,
        vlm_client: Optional[VLMClient] = None,
        rag_writer: Optional[CongestionRAGWriter] = None,
        confidence_threshold: float = 0.6
    ):
        """
        初始化拥堵分析器
        
        Args:
            vlm_client: VLM客户端，如未提供则创建默认客户端
            rag_writer: RAG写入器，如未提供则使用内存RAG
            confidence_threshold: 置信度阈值
        """
        self.vlm_client = vlm_client or VLMClient()
        self.rag_writer = rag_writer or CongestionRAGWriter()
        self.confidence_threshold = confidence_threshold
        
        # 活跃事件缓存
        self._active_events: Dict[str, CongestionEvent] = {}
        
        logger.info("CongestionAnalyzer初始化完成")
    
    def analyze_hotspot(
        self,
        hotspot: CongestionHotspot,
        image: np.ndarray
    ) -> CongestionEvent:
        """
        分析拥堵热点
        
        Args:
            hotspot: 拥堵热点数据
            image: 拥堵区域图像
            
        Returns:
            CongestionEvent: 拥堵事件
        """
        logger.info(f"开始分析拥堵热点: {hotspot.hotspot_id}")
        
        # 1. 使用VLM分析图像
        traffic_data = hotspot.to_traffic_data()
        analysis_result = self.vlm_client.analyze_congestion(image, traffic_data)
        
        # 2. 计算严重度
        severity = self._calculate_severity(hotspot, analysis_result)
        
        # 3. 创建拥堵事件
        event = CongestionEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            location=hotspot.location,
            camera_ids=hotspot.camera_ids,
            density=hotspot.density,
            avg_speed=hotspot.avg_speed,
            vehicle_types=hotspot.vehicle_types,
            duration=hotspot.duration,
            cause=analysis_result.cause,
            cause_confidence=analysis_result.confidence,
            cause_description=analysis_result.description,
            severity=severity,
            recommended_action=analysis_result.recommended_action,
            status="active"
        )
        
        # 4. 写入RAG
        success = self.rag_writer.write_event(event)
        if success:
            logger.info(f"拥堵事件已写入RAG: {event.event_id}")
        else:
            logger.warning(f"拥堵事件写入RAG失败: {event.event_id}")
        
        # 5. 缓存活跃事件
        self._active_events[event.event_id] = event
        
        return event
    
    def analyze_hotspot_batch(
        self,
        hotspots: List[CongestionHotspot],
        images: List[np.ndarray]
    ) -> List[CongestionEvent]:
        """
        批量分析拥堵热点
        
        Args:
            hotspots: 拥堵热点列表
            images: 对应的图像列表
            
        Returns:
            拥堵事件列表
        """
        if len(hotspots) != len(images):
            raise ValueError("热点数量和图像数量不匹配")
        
        events = []
        for hotspot, image in zip(hotspots, images):
            try:
                event = self.analyze_hotspot(hotspot, image)
                events.append(event)
            except Exception as e:
                logger.error(f"分析热点 {hotspot.hotspot_id} 失败: {e}")
                continue
        
        logger.info(f"批量分析完成: {len(events)}/{len(hotspots)} 成功")
        return events
    
    def _calculate_severity(
        self,
        hotspot: CongestionHotspot,
        analysis_result: CongestionAnalysisResult
    ) -> str:
        """
        计算拥堵严重度
        
        Args:
            hotspot: 拥堵热点
            analysis_result: VLM分析结果
            
        Returns:
            严重度等级 (low | medium | high | critical)
        """
        # 基于多个因素计算严重度分数
        score = 0.0
        
        # 1. 车辆密度权重 (0-0.3)
        density_score = min(hotspot.density / 2.0, 1.0) * 0.3
        score += density_score
        
        # 2. 速度权重 (0-0.3)
        # 速度越低，分数越高
        speed_score = max(0, (60 - hotspot.avg_speed) / 60) * 0.3
        score += speed_score
        
        # 3. 持续时间权重 (0-0.2)
        duration_score = min(hotspot.duration / 60, 1.0) * 0.2
        score += duration_score
        
        # 4. 原因置信度权重 (0-0.2)
        confidence_score = analysis_result.confidence * 0.2
        score += confidence_score
        
        # 确定严重度等级
        for severity, (low, high) in self.SEVERITY_THRESHOLDS.items():
            if low <= score < high:
                return severity
        
        return "critical"
    
    def resolve_event(self, event_id: str) -> bool:
        """
        标记拥堵事件为已解决
        
        Args:
            event_id: 事件ID
            
        Returns:
            是否成功
        """
        if event_id in self._active_events:
            event = self._active_events[event_id]
            event.status = "resolved"
            event.resolved_at = datetime.now()
            
            # 更新RAG
            success = self.rag_writer.write_event(event)
            
            if success:
                del self._active_events[event_id]
                logger.info(f"事件已解决: {event_id}")
            
            return success
        
        logger.warning(f"未找到事件: {event_id}")
        return False
    
    def get_active_events(self) -> List[CongestionEvent]:
        """
        获取所有活跃事件
        
        Returns:
            活跃事件列表
        """
        return list(self._active_events.values())
    
    def get_event_statistics(self) -> Dict[str, Any]:
        """
        获取事件统计信息
        
        Returns:
            统计信息字典
        """
        events = list(self._active_events.values())
        
        # 按原因统计
        cause_counts = {}
        for event in events:
            cause_counts[event.cause] = cause_counts.get(event.cause, 0) + 1
        
        # 按严重度统计
        severity_counts = {}
        for event in events:
            severity_counts[event.severity] = severity_counts.get(event.severity, 0) + 1
        
        return {
            "total_active": len(events),
            "by_cause": cause_counts,
            "by_severity": severity_counts,
            "rag_statistics": self.rag_writer.get_statistics()
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        健康检查
        
        Returns:
            健康状态字典
        """
        return {
            "vlm_client": self.vlm_client.health_check(),
            "rag_writer": {
                "backend_type": type(self.rag_writer.rag).__name__,
                "statistics": self.rag_writer.get_statistics()
            },
            "active_events_count": len(self._active_events)
        }
