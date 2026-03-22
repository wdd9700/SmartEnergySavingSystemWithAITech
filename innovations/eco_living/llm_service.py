"""
LLM对话服务

提供基于大语言模型的节能建议对话功能。
支持上下文理解和多轮对话。
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass, field
from datetime import datetime

from .rag_client import get_in_memory_rag, RAGQueryResult

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """聊天消息"""
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatSession:
    """聊天会话"""
    session_id: str
    messages: List[ChatMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)


class LLMService:
    """
    LLM对话服务
    
    集成RAG检索和大语言模型生成，提供智能对话能力。
    """
    
    SYSTEM_PROMPT = """你是一个专业的节能减排顾问，名为"绿色助手"。

你的职责是：
1. 根据用户的问题，提供具体、实用的节能减排建议
2. 结合知识库中的信息，给出准确的回答
3. 鼓励用户采取环保行动，培养绿色生活习惯
4. 用友好、易懂的语言与用户交流

回答原则：
- 优先使用知识库中的具体建议
- 根据用户场景提供个性化建议
- 说明建议的实施难度和预期效果
- 鼓励用户从小事做起，逐步养成环保习惯

当前时间：{current_time}
"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化LLM服务
        
        Args:
            api_key: LLM API密钥
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
        self.rag = get_in_memory_rag()
        
        # 会话管理
        self.sessions: Dict[str, ChatSession] = {}
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """
        创建新会话
        
        Args:
            session_id: 可选的会话ID
            
        Returns:
            会话ID
        """
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.sessions)}"
        
        self.sessions[session_id] = ChatSession(
            session_id=session_id,
            messages=[
                ChatMessage(
                    role="system",
                    content=self.SYSTEM_PROMPT.format(
                        current_time=datetime.now().isoformat()
                    )
                )
            ]
        )
        
        return session_id
    
    def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        处理用户消息
        
        Args:
            message: 用户消息
            session_id: 会话ID（None则创建新会话）
            stream: 是否流式返回
            
        Returns:
            回复内容
        """
        # 获取或创建会话
        if session_id is None or session_id not in self.sessions:
            session_id = self.create_session(session_id)
        
        session = self.sessions[session_id]
        
        # 添加用户消息
        session.messages.append(ChatMessage(role="user", content=message))
        
        # RAG检索
        rag_result = self.rag.query(message, top_k=5)
        
        # 生成回复
        if stream:
            return self._generate_stream(session, rag_result)
        else:
            return self._generate_response(session, rag_result)
    
    def _generate_response(
        self,
        session: ChatSession,
        rag_result: RAGQueryResult
    ) -> Dict[str, Any]:
        """生成完整回复"""
        # 构建提示
        context = self._build_context(session, rag_result)
        
        # 调用LLM或本地生成
        if self.api_key:
            response_text = self._call_llm_api(context)
        else:
            response_text = self._generate_local(rag_result)
        
        # 添加助手消息
        session.messages.append(
            ChatMessage(
                role="assistant",
                content=response_text,
                metadata={
                    "rag_confidence": rag_result.confidence,
                    "sources_count": len(rag_result.sources)
                }
            )
        )
        
        return {
            "session_id": session.session_id,
            "response": response_text,
            "sources": rag_result.sources,
            "confidence": rag_result.confidence,
            "message_count": len(session.messages)
        }
    
    def _generate_stream(
        self,
        session: ChatSession,
        rag_result: RAGQueryResult
    ) -> Generator[str, None, None]:
        """流式生成回复"""
        # 简化实现：先生成完整回复，再分段yield
        response = self._generate_response(session, rag_result)
        
        text = response["response"]
        # 按句子分割
        sentences = text.replace("。", "。\n").replace("！", "！\n").replace("？", "？\n").split("\n")
        
        for sentence in sentences:
            if sentence.strip():
                yield sentence.strip()
    
    def _build_context(
        self,
        session: ChatSession,
        rag_result: RAGQueryResult
    ) -> str:
        """构建LLM上下文"""
        # 获取最近的消息历史
        recent_messages = session.messages[-5:]  # 最近5条
        
        # 构建对话历史
        history_parts = []
        for msg in recent_messages:
            if msg.role == "user":
                history_parts.append(f"用户：{msg.content}")
            elif msg.role == "assistant":
                history_parts.append(f"助手：{msg.content}")
        
        # 构建RAG上下文
        rag_context = "\n\n".join([
            f"参考信息 {i+1}：{source['content']}"
            for i, source in enumerate(rag_result.sources[:3])
        ])
        
        # 组合完整提示
        prompt = f"""{self.SYSTEM_PROMPT}

对话历史：
{"\n".join(history_parts)}

检索到的相关信息：
{rag_context}

请基于以上信息，回答用户的问题。回答要具体、实用，并鼓励用户采取环保行动。
"""
        return prompt
    
    def _call_llm_api(self, prompt: str) -> str:
        """调用LLM API"""
        try:
            import openai
            
            client = openai.OpenAI(api_key=self.api_key)
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "请回答"}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM API调用失败: {e}")
            return self._generate_local_from_prompt(prompt)
    
    def _generate_local(self, rag_result: RAGQueryResult) -> str:
        """本地生成回复（不使用外部API）"""
        return rag_result.answer
    
    def _generate_local_from_prompt(self, prompt: str) -> str:
        """从提示生成本地回复"""
        # 提取用户问题
        lines = prompt.split("\n")
        user_question = ""
        for line in lines:
            if line.startswith("用户："):
                user_question = line[3:]
                break
        
        # 重新查询RAG
        if user_question:
            result = self.rag.query(user_question, top_k=5)
            return result.answer
        
        return "抱歉，我暂时无法回答这个问题。您可以尝试询问关于家庭节能、绿色出行或办公环保的建议。"
    
    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """获取会话历史"""
        if session_id not in self.sessions:
            return []
        
        session = self.sessions[session_id]
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat()
            }
            for msg in session.messages
            if msg.role != "system"
        ]
    
    def clear_session(self, session_id: str):
        """清空会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def get_suggested_questions(self) -> List[str]:
        """获取推荐问题"""
        return [
            "家庭生活中有哪些简单的节能方法？",
            "夏天怎么使用空调更省电？",
            "绿色出行有哪些选择？",
            "办公室里如何节约用电？",
            "怎样减少日常生活中的水资源浪费？",
            "购买家电时应该注意哪些能效指标？",
            "如何处理厨余垃圾更环保？",
            "有哪些减少塑料使用的好方法？"
        ]


class EcoAdvisor:
    """
    节能顾问（简化接口）
    
    提供简化的节能建议查询接口。
    """
    
    def __init__(self):
        self.llm = LLMService()
    
    def ask(self, question: str) -> str:
        """
        提问并获取回答
        
        Args:
            question: 问题
            
        Returns:
            回答文本
        """
        response = self.llm.chat(question)
        return response["response"]
    
    def get_tips_by_category(self, category: str) -> List[Dict]:
        """
        按类别获取建议
        
        Args:
            category: 类别名称
            
        Returns:
            建议列表
        """
        from .knowledge_base import get_knowledge_base
        
        kb = get_knowledge_base()
        
        # 类别映射
        category_map = {
            "家庭": "家庭生活",
            "出行": "出行方式",
            "办公": "办公学习",
            "饮食": "饮食消费",
            "社区": "社区参与"
        }
        
        cat_name = category_map.get(category, category)
        tips = kb.get_by_category(cat_name)
        
        return [
            {
                "title": tip.title,
                "content": tip.content,
                "difficulty": tip.difficulty,
                "impact": tip.impact,
                "tags": tip.tags
            }
            for tip in tips[:10]
        ]
    
    def get_daily_tip(self) -> Dict[str, str]:
        """获取每日一条建议"""
        import random
        from .knowledge_base import get_knowledge_base
        
        kb = get_knowledge_base()
        easy_tips = kb.get_by_difficulty("easy")
        
        if easy_tips:
            tip = random.choice(easy_tips)
            return {
                "title": tip.title,
                "content": tip.content,
                "category": tip.category,
                "action": "今天就可以尝试这样做！"
            }
        
        return {
            "title": "随手关灯",
            "content": "离开房间时随手关闭电灯和电器，这是最简单有效的节能习惯。",
            "category": "家庭生活",
            "action": "从现在开始养成习惯！"
        }


# 全局服务实例
_llm_service: Optional[LLMService] = None
_advisor: Optional[EcoAdvisor] = None


def get_llm_service() -> LLMService:
    """获取LLM服务单例"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


def get_advisor() -> EcoAdvisor:
    """获取节能顾问单例"""
    global _advisor
    if _advisor is None:
        _advisor = EcoAdvisor()
    return _advisor
