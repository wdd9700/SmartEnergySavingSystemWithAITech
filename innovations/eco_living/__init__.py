"""
生活节能减排智能助手

基于YouTu GraphRAG和LLM的生活节能建议系统。
提供智能对话、知识检索和节能建议功能。
"""

from .knowledge_base import EcoLivingKnowledgeBase, get_knowledge_base, EcoTip
from .rag_client import (
    YouTuGraphRAGClient,
    InMemoryGraphRAG,
    get_rag_client,
    get_in_memory_rag,
    RAGQueryResult
)
from .llm_service import (
    LLMService,
    EcoAdvisor,
    get_llm_service,
    get_advisor,
    ChatMessage,
    ChatSession
)

__version__ = "1.0.0"
__all__ = [
    # 知识库
    "EcoLivingKnowledgeBase",
    "get_knowledge_base",
    "EcoTip",
    
    # RAG客户端
    "YouTuGraphRAGClient",
    "InMemoryGraphRAG",
    "get_rag_client",
    "get_in_memory_rag",
    "RAGQueryResult",
    
    # LLM服务
    "LLMService",
    "EcoAdvisor",
    "get_llm_service",
    "get_advisor",
    "ChatMessage",
    "ChatSession",
]


def init_knowledge_base():
    """初始化知识库"""
    kb = get_knowledge_base()
    stats = kb.get_statistics()
    print(f"知识库初始化完成: {stats['total_tips']} 条建议")
    return kb


def quick_ask(question: str) -> str:
    """
    快速提问接口
    
    Args:
        question: 问题文本
        
    Returns:
        回答文本
    """
    advisor = get_advisor()
    return advisor.ask(question)
