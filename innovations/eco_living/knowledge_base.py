"""
生活节能减排知识库模块

基于YouTu GraphRAG的生活节能知识库系统。
将社会调查报告内容构建为知识图谱，支持智能问答。
"""

import os
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class EcoTip:
    """
    节能建议条目
    
    Attributes:
        category: 类别（家庭/出行/办公/饮食等）
        title: 标题
        content: 详细内容
        tags: 标签列表
        difficulty: 实施难度（easy/medium/hard）
        impact: 节能效果（low/medium/high）
    """
    category: str
    title: str
    content: str
    tags: List[str] = field(default_factory=list)
    difficulty: str = "easy"
    impact: str = "medium"


class EcoLivingKnowledgeBase:
    """
    生活节能知识库
    
    管理节能减排建议的存储、索引和检索
    """
    
    CATEGORIES = {
        "family": "家庭生活",
        "travel": "出行方式", 
        "office": "办公学习",
        "community": "社区参与",
        "diet": "饮食消费",
        "clothing": "服饰形象",
        "tourism": "旅游休闲",
        "building": "建筑装修",
        "industry": "工业农业",
        "policy": "政策倡导"
    }
    
    def __init__(self, storage_path: str = "data/eco_living_kb.json"):
        """
        初始化知识库
        
        Args:
            storage_path: 知识库存储路径
        """
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.tips: List[EcoTip] = []
        self.entities: Dict[str, Any] = {}
        self.relations: List[Dict] = []
        
        # 加载或初始化
        if self.storage_path.exists():
            self._load()
        else:
            self._init_from_document()
    
    def _init_from_document(self):
        """从社会调查报告初始化知识库"""
        doc_path = Path("docs/生活中的节能减排建议.md")
        if doc_path.exists():
            self._parse_document(doc_path.read_text(encoding='utf-8'))
            self._save()
    
    def _parse_document(self, content: str):
        """
        解析社会调查报告内容
        
        提取结构化的节能建议
        """
        # 按类别分割内容
        sections = self._split_sections(content)
        
        for category_key, category_name in self.CATEGORIES.items():
            if category_key in sections:
                section_content = sections[category_key]
                tips = self._extract_tips(section_content, category_name)
                self.tips.extend(tips)
        
        logger.info(f"从文档提取了 {len(self.tips)} 条节能建议")
        
        # 构建知识图谱
        self._build_knowledge_graph()
    
    def _split_sections(self, content: str) -> Dict[str, str]:
        """将文档按类别分割"""
        sections = {}
        
        # 类别关键词映射 - 匹配文档中的实际表述
        category_patterns = [
            ("family", "在家庭生活方面"),
            ("travel", "在出行方式方面"),
            ("office", "在办公学习方面"),
            ("community", "在社区公共参与方面"),
            ("diet", "在饮食方面"),
            ("clothing", "在个人形象和服饰方面"),
            ("tourism", "在旅游休闲方面"),
            ("building", "在建筑装修方面"),
            ("industry", "在工业农业方面"),
            ("policy", "在政策倡导方面"),
            ("tech", "在技术创新方面"),
            ("daily", "在日常细节方面"),
            ("psychology", "在心理观念方面"),
            ("pet", "在宠物饲养方面"),
            ("festival", "在节日庆祝方面"),
            ("agriculture", "在农业生产方面"),
            ("industry_prod", "在工业生产方面"),
            ("building_energy", "在建筑节能方面"),
            ("transport", "在交通基础设施方面"),
            ("research", "在科学技术研究方面"),
            ("education", "在宣传教育方面"),
        ]
        
        # 按类别分割
        for i, (cat_key, pattern) in enumerate(category_patterns):
            start_idx = content.find(pattern)
            if start_idx == -1:
                continue
                
            # 找到下一个类别的起始位置
            end_idx = len(content)
            for j in range(i + 1, len(category_patterns)):
                next_pattern = category_patterns[j][1]
                next_idx = content.find(next_pattern, start_idx + len(pattern))
                if next_idx != -1:
                    end_idx = next_idx
                    break
            
            # 提取内容
            section_content = content[start_idx:end_idx].strip()
            sections[cat_key] = section_content
        
        return sections
    
    def _extract_tips(self, content: str, category: str) -> List[EcoTip]:
        """从类别内容中提取具体建议"""
        tips = []
        
        # 按句子分割，提取包含节能关键词的建议
        sentences = content.replace('。', '\n').replace('；', '\n').split('\n')
        
        eco_keywords = [
            "节能", "减排", "省电", "节水", "省气", "省油",
            "减少", "降低", "节约", "环保", "绿色", "低碳",
            "循环利用", "重复使用", "回收", "可持续"
        ]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            
            # 检查是否包含节能关键词
            if any(keyword in sentence for keyword in eco_keywords):
                # 生成标题（前20字）
                title = sentence[:20] + "..." if len(sentence) > 20 else sentence
                
                # 提取标签
                tags = [kw for kw in eco_keywords if kw in sentence][:3]
                
                # 判断难度和效果
                difficulty = self._estimate_difficulty(sentence)
                impact = self._estimate_impact(sentence)
                
                tip = EcoTip(
                    category=category,
                    title=title,
                    content=sentence,
                    tags=tags,
                    difficulty=difficulty,
                    impact=impact
                )
                tips.append(tip)
        
        return tips
    
    def _estimate_difficulty(self, content: str) -> str:
        """估计实施难度"""
        easy_keywords = ["随手", "关闭", "拔掉", "减少", "选择"]
        hard_keywords = ["安装", "改造", "装修", "建设", "系统"]
        
        if any(kw in content for kw in easy_keywords):
            return "easy"
        elif any(kw in content for kw in hard_keywords):
            return "hard"
        return "medium"
    
    def _estimate_impact(self, content: str) -> str:
        """估计节能效果"""
        high_keywords = ["太阳能", "光伏", "新能源", "系统", "50%", "一半"]
        low_keywords = ["少量", "偶尔", "一次"]
        
        if any(kw in content for kw in high_keywords):
            return "high"
        elif any(kw in content for kw in low_keywords):
            return "low"
        return "medium"
    
    def _build_knowledge_graph(self):
        """构建知识图谱"""
        # 提取实体
        for tip in self.tips:
            # 类别实体
            if tip.category not in self.entities:
                self.entities[tip.category] = {
                    "type": "category",
                    "name": tip.category,
                    "tips_count": 0
                }
            self.entities[tip.category]["tips_count"] += 1
            
            # 标签实体
            for tag in tip.tags:
                if tag not in self.entities:
                    self.entities[tag] = {
                        "type": "tag",
                        "name": tag,
                        "tips": []
                    }
                self.entities[tag]["tips"].append(tip.title)
            
            # 创建关系
            self.relations.append({
                "source": tip.category,
                "target": tip.title,
                "type": "contains"
            })
            
            for tag in tip.tags:
                self.relations.append({
                    "source": tip.title,
                    "target": tag,
                    "type": "has_tag"
                })
    
    def search(self, query: str, top_k: int = 5) -> List[EcoTip]:
        """
        搜索相关节能建议
        
        Args:
            query: 查询关键词
            top_k: 返回结果数量
            
        Returns:
            相关建议列表
        """
        query_lower = query.lower()
        scores = []
        
        for tip in self.tips:
            score = 0
            
            # 标题匹配
            if query_lower in tip.title.lower():
                score += 3
            
            # 内容匹配
            if query_lower in tip.content.lower():
                score += 2
            
            # 标签匹配
            if any(query_lower in tag.lower() for tag in tip.tags):
                score += 2
            
            # 类别匹配
            if query_lower in tip.category.lower():
                score += 1
            
            if score > 0:
                scores.append((tip, score))
        
        # 按分数排序
        scores.sort(key=lambda x: x[1], reverse=True)
        return [tip for tip, _ in scores[:top_k]]
    
    def get_by_category(self, category: str) -> List[EcoTip]:
        """按类别获取建议"""
        return [tip for tip in self.tips if tip.category == category]
    
    def get_by_difficulty(self, difficulty: str) -> List[EcoTip]:
        """按难度获取建议"""
        return [tip for tip in self.tips if tip.difficulty == difficulty]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        return {
            "total_tips": len(self.tips),
            "categories": {
                cat: len(self.get_by_category(cat))
                for cat in self.CATEGORIES.values()
            },
            "difficulty_distribution": {
                "easy": len(self.get_by_difficulty("easy")),
                "medium": len(self.get_by_difficulty("medium")),
                "hard": len(self.get_by_difficulty("hard"))
            },
            "entity_count": len(self.entities),
            "relation_count": len(self.relations)
        }
    
    def _save(self):
        """保存知识库到文件"""
        data = {
            "tips": [
                {
                    "category": tip.category,
                    "title": tip.title,
                    "content": tip.content,
                    "tags": tip.tags,
                    "difficulty": tip.difficulty,
                    "impact": tip.impact
                }
                for tip in self.tips
            ],
            "entities": self.entities,
            "relations": self.relations,
            "updated_at": datetime.now().isoformat()
        }
        
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _load(self):
        """从文件加载知识库"""
        with open(self.storage_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.tips = [
            EcoTip(
                category=tip["category"],
                title=tip["title"],
                content=tip["content"],
                tags=tip["tags"],
                difficulty=tip["difficulty"],
                impact=tip["impact"]
            )
            for tip in data.get("tips", [])
        ]
        
        self.entities = data.get("entities", {})
        self.relations = data.get("relations", [])


# 全局知识库实例
_kb_instance: Optional[EcoLivingKnowledgeBase] = None


def get_knowledge_base() -> EcoLivingKnowledgeBase:
    """获取知识库单例"""
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = EcoLivingKnowledgeBase()
    return _kb_instance
