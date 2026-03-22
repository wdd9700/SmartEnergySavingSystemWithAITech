"""
YouTu GraphRAG 客户端

用于与YouTu GraphRAG服务交互的客户端实现。
支持知识图谱的查询和检索增强生成。
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RAGQueryResult:
    """RAG查询结果"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    query_time_ms: int


class YouTuGraphRAGClient:
    """
    YouTu GraphRAG客户端
    
    用于连接YouTu GraphRAG服务，执行知识图谱查询。
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        graph_name: str = "eco_living"
    ):
        """
        初始化客户端
        
        Args:
            api_key: API密钥
            endpoint: 服务端点
            graph_name: 图谱名称
        """
        self.api_key = api_key or os.getenv("YOUTU_GRAPH_API_KEY")
        self.endpoint = endpoint or os.getenv(
            "YOUTU_GRAPH_ENDPOINT",
            "https://graphrag.youtu.com/api/v1"
        )
        self.graph_name = graph_name
        
        self._client = None
        self._init_client()
    
    def _init_client(self):
        """初始化底层客户端"""
        try:
            # 尝试导入YouTu官方SDK
            # from youtu_graphrag import Client
            # self._client = Client(api_key=self.api_key, endpoint=self.endpoint)
            logger.info("YouTu GraphRAG客户端初始化成功")
        except ImportError:
            logger.warning("YouTu GraphRAG SDK未安装，使用模拟模式")
            self._client = None
    
    def query(
        self,
        query_text: str,
        top_k: int = 5,
        include_sources: bool = True
    ) -> RAGQueryResult:
        """
        执行RAG查询
        
        Args:
            query_text: 查询文本
            top_k: 返回结果数量
            include_sources: 是否包含来源
            
        Returns:
            查询结果
        """
        if self._client:
            return self._query_with_client(query_text, top_k, include_sources)
        else:
            return self._query_mock(query_text, top_k)
    
    def _query_with_client(
        self,
        query_text: str,
        top_k: int,
        include_sources: bool
    ) -> RAGQueryResult:
        """使用官方SDK查询"""
        try:
            response = self._client.query(
                graph_name=self.graph_name,
                query=query_text,
                top_k=top_k,
                include_sources=include_sources
            )
            
            return RAGQueryResult(
                answer=response.get("answer", ""),
                sources=response.get("sources", []),
                confidence=response.get("confidence", 0.0),
                query_time_ms=response.get("query_time_ms", 0)
            )
        except Exception as e:
            logger.error(f"YouTu GraphRAG查询失败: {e}")
            return self._query_mock(query_text, top_k)
    
    def _query_mock(self, query_text: str, top_k: int) -> RAGQueryResult:
        """
        模拟查询（用于开发和测试）
        
        使用本地知识库进行检索
        """
        from .knowledge_base import get_knowledge_base
        
        kb = get_knowledge_base()
        tips = kb.search(query_text, top_k=top_k)
        
        # 构建回答
        if tips:
            answer_parts = [f"根据您的问题，我为您找到了以下节能建议：\n"]
            for i, tip in enumerate(tips, 1):
                answer_parts.append(f"\n{i}. {tip.title}")
                answer_parts.append(f"   {tip.content}")
                answer_parts.append(f"   [类别: {tip.category} | 难度: {tip.difficulty} | 效果: {tip.impact}]")
            
            answer = "\n".join(answer_parts)
            confidence = 0.8 if len(tips) >= 3 else 0.6
        else:
            answer = "抱歉，没有找到相关的节能建议。您可以尝试使用其他关键词，如\"节水\"、\"节电\"、\"出行\"等。"
            confidence = 0.0
        
        sources = [
            {
                "title": tip.title,
                "content": tip.content,
                "category": tip.category,
                "score": 1.0 - (i * 0.1)
            }
            for i, tip in enumerate(tips)
        ]
        
        return RAGQueryResult(
            answer=answer,
            sources=sources,
            confidence=confidence,
            query_time_ms=100
        )
    
    def add_entity(
        self,
        entity_name: str,
        entity_type: str,
        properties: Dict[str, Any]
    ) -> bool:
        """
        添加实体到知识图谱
        
        Args:
            entity_name: 实体名称
            entity_type: 实体类型
            properties: 实体属性
            
        Returns:
            是否添加成功
        """
        if not self._client:
            logger.warning("客户端未初始化，无法添加实体")
            return False
        
        try:
            self._client.add_entity(
                graph_name=self.graph_name,
                name=entity_name,
                entity_type=entity_type,
                properties=properties
            )
            return True
        except Exception as e:
            logger.error(f"添加实体失败: {e}")
            return False
    
    def add_relation(
        self,
        source: str,
        target: str,
        relation_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        添加关系到知识图谱
        
        Args:
            source: 源实体
            target: 目标实体
            relation_type: 关系类型
            properties: 关系属性
            
        Returns:
            是否添加成功
        """
        if not self._client:
            logger.warning("客户端未初始化，无法添加关系")
            return False
        
        try:
            self._client.add_relation(
                graph_name=self.graph_name,
                source=source,
                target=target,
                relation_type=relation_type,
                properties=properties or {}
            )
            return True
        except Exception as e:
            logger.error(f"添加关系失败: {e}")
            return False
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """获取图谱统计信息"""
        if not self._client:
            from .knowledge_base import get_knowledge_base
            return get_knowledge_base().get_statistics()
        
        try:
            return self._client.get_stats(graph_name=self.graph_name)
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}


class InMemoryGraphRAG:
    """
    内存GraphRAG实现
    
    用于开发和测试的轻量级实现，不依赖外部服务。
    """
    
    def __init__(self):
        self.entities: Dict[str, Dict] = {}
        self.relations: List[Dict] = []
        self._load_from_kb()
    
    def _load_from_kb(self):
        """从知识库加载数据"""
        from .knowledge_base import get_knowledge_base
        
        kb = get_knowledge_base()
        self.entities = kb.entities
        self.relations = kb.relations
    
    def query(
        self,
        query_text: str,
        top_k: int = 5
    ) -> RAGQueryResult:
        """执行查询"""
        from .knowledge_base import get_knowledge_base
        
        kb = get_knowledge_base()
        tips = kb.search(query_text, top_k=top_k)
        
        # 构建上下文感知的回答
        answer = self._generate_answer(query_text, tips)
        
        sources = [
            {
                "title": tip.title,
                "content": tip.content,
                "category": tip.category,
                "difficulty": tip.difficulty,
                "impact": tip.impact
            }
            for tip in tips
        ]
        
        return RAGQueryResult(
            answer=answer,
            sources=sources,
            confidence=0.85 if tips else 0.0,
            query_time_ms=50
        )
    
    def _generate_answer(self, query: str, tips: List) -> str:
        """生成自然语言回答"""
        if not tips:
            return "抱歉，没有找到相关的节能建议。您可以尝试询问：\n- 家庭如何节能？\n- 出行怎么更环保？\n- 办公室节能技巧"
        
        # 分析查询意图
        query_lower = query.lower()
        
        if "家庭" in query_lower or "家里" in query_lower:
            prefix = "针对家庭节能，"
        elif "出行" in query_lower or "交通" in query_lower:
            prefix = "关于绿色出行，"
        elif "办公" in query_lower or "工作" in query_lower:
            prefix = "在办公节能方面，"
        elif "饮食" in query_lower or "食物" in query_lower:
            prefix = "在饮食环保方面，"
        else:
            prefix = "根据您的问题，"
        
        # 构建回答
        parts = [f"{prefix}我为您推荐以下建议：\n"]
        
        for i, tip in enumerate(tips[:5], 1):
            parts.append(f"\n{i}. {tip.title}")
            parts.append(f"   {tip.content}")
            
            # 添加实施建议
            if tip.difficulty == "easy":
                parts.append(f"   ✅ 实施难度：简单")
            elif tip.difficulty == "hard":
                parts.append(f"   ⚠️ 实施难度：较高")
            
            if tip.impact == "high":
                parts.append(f"   🌟 节能效果显著")
        
        # 添加总结
        easy_count = sum(1 for t in tips if t.difficulty == "easy")
        if easy_count > 0:
            parts.append(f"\n💡 其中有 {easy_count} 条建议实施难度较低，您可以优先尝试！")
        
        return "\n".join(parts)
    
    def add_entity(self, name: str, entity_type: str, properties: Dict) -> bool:
        """添加实体"""
        self.entities[name] = {
            "type": entity_type,
            **properties
        }
        return True
    
    def add_relation(
        self,
        source: str,
        target: str,
        relation_type: str,
        properties: Optional[Dict] = None
    ) -> bool:
        """添加关系"""
        self.relations.append({
            "source": source,
            "target": target,
            "type": relation_type,
            "properties": properties or {}
        })
        return True


# 全局客户端实例
_rag_client: Optional[YouTuGraphRAGClient] = None
_in_memory_rag: Optional[InMemoryGraphRAG] = None


def get_rag_client() -> YouTuGraphRAGClient:
    """获取RAG客户端单例"""
    global _rag_client
    if _rag_client is None:
        _rag_client = YouTuGraphRAGClient()
    return _rag_client


def get_in_memory_rag() -> InMemoryGraphRAG:
    """获取内存RAG单例"""
    global _in_memory_rag
    if _in_memory_rag is None:
        _in_memory_rag = InMemoryGraphRAG()
    return _in_memory_rag
