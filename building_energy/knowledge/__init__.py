"""
知识库模块 (Knowledge Base Module)

提供基于GraphRAG的建筑能耗知识库问答系统。

主要组件:
- DocumentLoader: 文档加载和解析
- KnowledgeBase: 知识库核心类，提供查询和优化建议功能
"""

from .document_loader import DocumentLoader, Document
from .graph_rag import KnowledgeBase

__all__ = [
    "DocumentLoader",
    "Document",
    "KnowledgeBase",
]
