"""
GraphRAG知识库模块

基于GraphRAG的建筑能耗知识库问答系统。
支持向量检索和图结构查询，提供能耗优化建议。
"""

import os
import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime

import numpy as np

from .document_loader import DocumentLoader, Document, DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """
    查询结果数据结构
    
    Attributes:
        answer: 答案文本
        sources: 来源文档列表
        confidence: 置信度分数
        metadata: 额外元数据
    """
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Entity:
    """
    知识图谱实体
    
    Attributes:
        name: 实体名称
        entity_type: 实体类型
        properties: 实体属性
        embedding: 实体向量表示
    """
    name: str
    entity_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


@dataclass
class Relation:
    """
    知识图谱关系
    
    Attributes:
        source: 源实体名称
        target: 目标实体名称
        relation_type: 关系类型
        properties: 关系属性
    """
    source: str
    target: str
    relation_type: str
    properties: Dict[str, Any] = field(default_factory=dict)


class EmbeddingProvider:
    """
    向量嵌入提供器
    
    使用sentence-transformers生成文本向量
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        初始化嵌入提供器
        
        Args:
            model_name: 模型名称
        """
        self.model_name = model_name
        self._model = None
        self._embedding_dim = 384  # MiniLM默认维度
    
    @property
    def model(self):
        """懒加载模型"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                self._embedding_dim = self._model.get_sentence_embedding_dimension()
            except ImportError:
                logger.warning("sentence-transformers not installed, using fallback")
                self._model = "fallback"
        return self._model
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        生成文本向量
        
        Args:
            texts: 文本列表
            
        Returns:
            向量数组 (n_texts, embedding_dim)
        """
        if self.model == "fallback":
            # 回退：使用简单的哈希向量
            return self._fallback_embed(texts)
        
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return np.array(embeddings)
    
    def _fallback_embed(self, texts: List[str]) -> np.ndarray:
        """回退嵌入方法（简单哈希）"""
        embeddings = []
        for text in texts:
            # 使用文本哈希生成伪向量
            hash_val = hash(text) % (2**32)
            np.random.seed(hash_val)
            embedding = np.random.randn(self._embedding_dim)
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
        return np.array(embeddings)
    
    @property
    def embedding_dim(self) -> int:
        """获取嵌入维度"""
        _ = self.model  # 确保模型已加载
        return self._embedding_dim


class VectorStore:
    """
    向量存储
    
    使用FAISS或简单余弦相似度进行向量检索
    """
    
    def __init__(self, embedding_dim: int, use_faiss: bool = True):
        """
        初始化向量存储
        
        Args:
            embedding_dim: 向量维度
            use_faiss: 是否使用FAISS（如果可用）
        """
        self.embedding_dim = embedding_dim
        self.use_faiss = use_faiss and self._check_faiss()
        
        self._vectors: List[np.ndarray] = []
        self._metadata: List[Dict[str, Any]] = []
        self._faiss_index = None
    
    def _check_faiss(self) -> bool:
        """检查FAISS是否可用"""
        try:
            import faiss
            return True
        except ImportError:
            return False
    
    def add(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """
        添加向量
        
        Args:
            vectors: 向量数组 (n, embedding_dim)
            metadata: 元数据列表
        """
        for i, vec in enumerate(vectors):
            self._vectors.append(vec)
            self._metadata.append(metadata[i] if i < len(metadata) else {})
        
        # 重建FAISS索引
        if self.use_faiss:
            self._build_faiss_index()
    
    def _build_faiss_index(self) -> None:
        """构建FAISS索引"""
        if not self._vectors:
            return
        
        import faiss
        
        vectors_array = np.array(self._vectors).astype('float32')
        
        # 使用IndexFlatIP进行余弦相似度（归一化后）
        self._faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        
        # 归一化向量
        faiss.normalize_L2(vectors_array)
        self._faiss_index.add(vectors_array)
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        搜索最相似的向量
        
        Args:
            query_vector: 查询向量
            top_k: 返回结果数量
            
        Returns:
            (元数据, 相似度分数)列表
        """
        if not self._vectors:
            return []
        
        if self.use_faiss and self._faiss_index is not None:
            return self._faiss_search(query_vector, top_k)
        else:
            return self._naive_search(query_vector, top_k)
    
    def _faiss_search(
        self,
        query_vector: np.ndarray,
        top_k: int
    ) -> List[Tuple[Dict[str, Any], float]]:
        """使用FAISS搜索"""
        import faiss
        
        query = query_vector.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query)
        
        scores, indices = self._faiss_index.search(query, min(top_k, len(self._vectors)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((self._metadata[idx], float(score)))
        
        return results
    
    def _naive_search(
        self,
        query_vector: np.ndarray,
        top_k: int
    ) -> List[Tuple[Dict[str, Any], float]]:
        """使用朴素余弦相似度搜索"""
        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)
        
        similarities = []
        for vec, meta in zip(self._vectors, self._metadata):
            vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
            sim = np.dot(query_norm, vec_norm)
            similarities.append((meta, sim))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def save(self, path: str) -> None:
        """保存向量存储"""
        data = {
            'vectors': [v.tolist() for v in self._vectors],
            'metadata': self._metadata,
            'embedding_dim': self.embedding_dim
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
    
    def load(self, path: str) -> None:
        """加载向量存储"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self._vectors = [np.array(v) for v in data['vectors']]
        self._metadata = data['metadata']
        self.embedding_dim = data['embedding_dim']
        
        if self.use_faiss:
            self._build_faiss_index()


class KnowledgeGraph:
    """
    知识图谱
    
    存储实体和关系，支持图查询
    """
    
    def __init__(self):
        """初始化知识图谱"""
        self._entities: Dict[str, Entity] = {}
        self._relations: List[Relation] = []
        self._entity_relations: Dict[str, List[Relation]] = {}  # 实体 -> 关系索引
    
    def add_entity(self, entity: Entity) -> None:
        """添加实体"""
        self._entities[entity.name] = entity
        if entity.name not in self._entity_relations:
            self._entity_relations[entity.name] = []
    
    def add_relation(self, relation: Relation) -> None:
        """添加关系"""
        self._relations.append(relation)
        
        # 更新索引
        if relation.source in self._entity_relations:
            self._entity_relations[relation.source].append(relation)
        if relation.target in self._entity_relations:
            self._entity_relations[relation.target].append(relation)
    
    def get_entity(self, name: str) -> Optional[Entity]:
        """获取实体"""
        return self._entities.get(name)
    
    def get_related_entities(
        self,
        entity_name: str,
        relation_type: Optional[str] = None
    ) -> List[Tuple[Entity, str]]:
        """
        获取相关实体
        
        Args:
            entity_name: 实体名称
            relation_type: 关系类型过滤
            
        Returns:
            (相关实体, 关系类型)列表
        """
        results = []
        relations = self._entity_relations.get(entity_name, [])
        
        for rel in relations:
            if relation_type and rel.relation_type != relation_type:
                continue
            
            if rel.source == entity_name:
                target = self._entities.get(rel.target)
                if target:
                    results.append((target, rel.relation_type))
            elif rel.target == entity_name:
                source = self._entities.get(rel.source)
                if source:
                    results.append((source, rel.relation_type))
        
        return results
    
    def extract_from_text(self, text: str) -> Tuple[List[Entity], List[Relation]]:
        """
        从文本中提取实体和关系（简化版）
        
        实际应用中可以使用spaCy或LLM进行更精确的提取
        """
        entities = []
        relations = []
        
        # 定义建筑能耗领域的实体模式
        entity_patterns = {
            'HVAC_SYSTEM': [
                r'空调系统', r'HVAC', r'中央空调', r'分体空调', r'风机盘管',
                r'chiller', r'air conditioner', r'cooling tower'
            ],
            'SENSOR': [
                r'温度传感器', r'湿度传感器', r'CO2传感器', r'光照传感器',
                r'temperature sensor', r'humidity sensor'
            ],
            'ENERGY': [
                r'能耗', r'电力', r'电能', r'功率', r'energy consumption',
                r'power consumption', r'electricity'
            ],
            'CONTROL': [
                r'控制策略', r'PID控制', r'模型预测控制', r'MPC',
                r'control strategy', r'PID controller'
            ],
            'BUILDING': [
                r'建筑', r'房间', r'楼层', r'办公楼', r'教室',
                r'building', r'room', r'floor', r'office'
            ]
        }
        
        # 简单实体提取
        found_entities = set()
        for entity_type, patterns in entity_patterns.items():
            for pattern in patterns:
                if pattern.lower() in text.lower():
                    found_entities.add((pattern, entity_type))
        
        for name, entity_type in found_entities:
            entity = Entity(
                name=name,
                entity_type=entity_type,
                properties={'source_text': text[:200]}
            )
            entities.append(entity)
        
        return entities, relations
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'entities': [
                {
                    'name': e.name,
                    'type': e.entity_type,
                    'properties': e.properties
                }
                for e in self._entities.values()
            ],
            'relations': [
                {
                    'source': r.source,
                    'target': r.target,
                    'type': r.relation_type,
                    'properties': r.properties
                }
                for r in self._relations
            ]
        }
    
    def save(self, path: str) -> None:
        """保存知识图谱"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    def load(self, path: str) -> None:
        """加载知识图谱"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self._entities.clear()
        self._relations.clear()
        self._entity_relations.clear()
        
        for e_data in data.get('entities', []):
            entity = Entity(
                name=e_data['name'],
                entity_type=e_data['type'],
                properties=e_data.get('properties', {})
            )
            self.add_entity(entity)
        
        for r_data in data.get('relations', []):
            relation = Relation(
                source=r_data['source'],
                target=r_data['target'],
                relation_type=r_data['type'],
                properties=r_data.get('properties', {})
            )
            self.add_relation(relation)


class KnowledgeBase:
    """
    知识库核心类
    
    基于GraphRAG的建筑能耗知识库，提供查询和优化建议功能。
    
    Attributes:
        root_dir: 知识库根目录
        document_loader: 文档加载器
        embedding_provider: 向量嵌入提供器
        vector_store: 向量存储
        knowledge_graph: 知识图谱
    
    Example:
        >>> kb = KnowledgeBase("./knowledge_base")
        >>> kb.add_document("docs/hvac_guide.md")
        >>> kb.index()
        >>> result = kb.query("如何优化空调能耗？")
        >>> print(result.answer)
    """
    
    def __init__(self, root_dir: str):
        """
        初始化知识库
        
        Args:
            root_dir: 知识库根目录（用于存储索引和缓存）
        """
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        
        # 子目录
        self._docs_dir = self.root_dir / "documents"
        self._docs_dir.mkdir(exist_ok=True)
        self._index_dir = self.root_dir / "index"
        self._index_dir.mkdir(exist_ok=True)
        
        # 组件初始化
        self.document_loader = DocumentLoader(chunk_size=1000, chunk_overlap=200)
        self.embedding_provider = EmbeddingProvider()
        self.vector_store: Optional[VectorStore] = None
        self.knowledge_graph = KnowledgeGraph()
        
        # 文档和块存储
        self._documents: Dict[str, Document] = {}
        self._chunks: Dict[str, DocumentChunk] = {}
        
        # 是否已索引
        self._indexed = False
        
        # 尝试加载已有索引
        self._load_index()
    
    def add_document(self, doc_path: str) -> None:
        """
        添加文档到知识库
        
        Args:
            doc_path: 文档路径
        """
        doc = self.document_loader.load(doc_path)
        self._documents[doc.doc_id] = doc
        
        # 复制到知识库文档目录
        import shutil
        dest_path = self._docs_dir / Path(doc_path).name
        shutil.copy2(doc_path, dest_path)
        
        logger.info(f"Added document: {doc_path} (ID: {doc.doc_id})")
        
        # 标记需要重新索引
        self._indexed = False
    
    def add_documents(self, doc_paths: List[str]) -> None:
        """批量添加文档"""
        for path in doc_paths:
            self.add_document(path)
    
    def index(self) -> None:
        """
        构建知识库索引
        
        包括：
        1. 文档分块
        2. 向量嵌入生成
        3. 向量存储构建
        4. 知识图谱构建
        """
        if not self._documents:
            logger.warning("No documents to index")
            return
        
        logger.info("Starting knowledge base indexing...")
        
        # 1. 文档分块
        all_chunks = []
        for doc in self._documents.values():
            chunks = self.document_loader.chunk_document(doc)
            all_chunks.extend(chunks)
            for chunk in chunks:
                self._chunks[chunk.chunk_id] = chunk
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(self._documents)} documents")
        
        # 2. 生成向量嵌入
        chunk_texts = [chunk.content for chunk in all_chunks]
        embeddings = self.embedding_provider.embed(chunk_texts)
        
        # 3. 构建向量存储
        self.vector_store = VectorStore(
            embedding_dim=self.embedding_provider.embedding_dim,
            use_faiss=True
        )
        
        metadata = [
            {
                'chunk_id': chunk.chunk_id,
                'doc_id': chunk.doc_id,
                'content': chunk.content,
                'metadata': chunk.metadata
            }
            for chunk in all_chunks
        ]
        
        self.vector_store.add(embeddings, metadata)
        
        # 4. 构建知识图谱
        for chunk in all_chunks:
            entities, relations = self.knowledge_graph.extract_from_text(chunk.content)
            for entity in entities:
                # 为实体生成嵌入
                entity.embedding = self.embedding_provider.embed([entity.name])[0]
                self.knowledge_graph.add_entity(entity)
            for relation in relations:
                self.knowledge_graph.add_relation(relation)
        
        self._indexed = True
        
        # 保存索引
        self._save_index()
        
        logger.info("Indexing completed successfully")
    
    def query(self, question: str, top_k: int = 5) -> QueryResult:
        """
        查询知识库
        
        Args:
            question: 问题文本
            top_k: 返回的相关文档数量
            
        Returns:
            QueryResult对象
        """
        if not self._indexed or self.vector_store is None:
            return QueryResult(
                answer="知识库尚未索引，请先调用index()方法构建索引。",
                confidence=0.0
            )
        
        # 1. 生成问题向量
        query_embedding = self.embedding_provider.embed([question])[0]
        
        # 2. 向量检索
        search_results = self.vector_store.search(query_embedding, top_k=top_k)
        
        if not search_results:
            return QueryResult(
                answer="抱歉，知识库中没有找到相关信息。",
                confidence=0.0
            )
        
        # 3. 构建答案
        contexts = []
        sources = []
        
        for meta, score in search_results:
            content = meta.get('content', '')
            chunk_metadata = meta.get('metadata', {})
            
            contexts.append(content)
            sources.append({
                'doc_id': meta.get('doc_id'),
                'chunk_id': meta.get('chunk_id'),
                'title': chunk_metadata.get('title', 'Unknown'),
                'file_name': chunk_metadata.get('file_name', 'Unknown'),
                'similarity': score
            })
        
        # 4. 生成答案（简化版，实际可以使用LLM）
        answer = self._generate_answer(question, contexts)
        
        # 5. 计算置信度
        confidence = sum(s['similarity'] for s in sources) / len(sources) if sources else 0
        
        return QueryResult(
            answer=answer,
            sources=sources,
            confidence=confidence,
            metadata={
                'query': question,
                'top_k': top_k,
                'timestamp': datetime.now().isoformat()
            }
        )
    
    def _generate_answer(self, question: str, contexts: List[str]) -> str:
        """
        基于检索到的上下文生成答案
        
        实现基于模板和关键词匹配的智能回答生成。
        支持建筑能耗领域的常见问题类型。
        
        Args:
            question: 用户问题
            contexts: 检索到的相关上下文列表
            
        Returns:
            生成的答案文本
        """
        if not contexts:
            return "抱歉，知识库中没有找到相关信息。"
        
        # 分析问题类型
        question_lower = question.lower()
        
        # 定义问题类型和关键词
        question_types = {
            'temperature': ['温度', '温控', '制冷', '制热', '设定温度', '空调温度', 'temperature'],
            'energy_saving': ['节能', '省电', '降低能耗', '优化', 'saving', 'optimization'],
            'hvac': ['空调', 'hvac', '中央空调', '风机', 'chiller', 'air conditioner'],
            'lighting': ['照明', '灯光', 'lighting', '灯具', 'led'],
            'control': ['控制', 'pid', 'mpc', '策略', 'control strategy'],
            'maintenance': ['维护', '保养', '清洗', '维护', 'maintenance'],
        }
        
        # 检测问题类型
        detected_types = []
        for qtype, keywords in question_types.items():
            if any(kw in question_lower for kw in keywords):
                detected_types.append(qtype)
        
        # 提取关键句子（基于简单评分）
        def score_sentence(sentence: str, question: str) -> float:
            """计算句子与问题的相关性分数"""
            score = 0.0
            question_words = set(question_lower.split())
            sentence_words = set(sentence.lower().split())
            
            # 关键词重叠
            overlap = question_words & sentence_words
            score += len(overlap) * 0.5
            
            # 数字和度量单位加分（通常包含具体建议）
            if any(c.isdigit() for c in sentence):
                score += 1.0
            
            # 建议性词汇加分
            advice_words = ['建议', '推荐', '应该', '可以', '优化', '设置', '调节']
            if any(w in sentence for w in advice_words):
                score += 0.5
            
            return score
        
        # 从上下文中提取关键句子
        key_points = []
        for ctx in contexts[:3]:  # 前3个上下文
            sentences = ctx.split('。')
            for sent in sentences:
                sent = sent.strip()
                if len(sent) > 10:  # 过滤短句
                    score = score_sentence(sent, question)
                    if score > 0.5:  # 只保留相关句子
                        key_points.append((sent, score))
        
        # 按分数排序并去重
        key_points.sort(key=lambda x: x[1], reverse=True)
        seen = set()
        unique_points = []
        for point, score in key_points[:5]:  # 最多5个要点
            # 简单去重：检查是否与已选要点相似
            is_duplicate = False
            for seen_point in seen:
                # 如果重叠度超过50%认为是重复
                overlap = set(point.split()) & set(seen_point.split())
                if len(overlap) / max(len(set(point.split())), 1) > 0.5:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen.add(point)
                unique_points.append(point)
        
        # 生成答案
        answer_parts = []
        
        # 开头
        if detected_types:
            type_names = {
                'temperature': '温度控制',
                'energy_saving': '节能优化',
                'hvac': 'HVAC系统',
                'lighting': '照明系统',
                'control': '控制策略',
                'maintenance': '设备维护'
            }
            type_str = '、'.join([type_names.get(t, t) for t in detected_types[:2]])
            answer_parts.append(f"关于{type_str}，为您提供以下信息：\n")
        else:
            answer_parts.append("根据知识库检索结果：\n")
        
        # 关键要点
        if unique_points:
            answer_parts.append("主要建议：")
            for i, point in enumerate(unique_points, 1):
                # 确保句子以句号结尾
                if not point.endswith('。') and not point.endswith('！'):
                    point += '。'
                answer_parts.append(f"{i}. {point}")
        else:
            # 如果没有提取到关键句子，使用原始上下文
            answer_parts.append("相关参考信息：")
            for i, ctx in enumerate(contexts[:3], 1):
                summary = ctx[:150] + "..." if len(ctx) > 150 else ctx
                answer_parts.append(f"{i}. {summary}")
        
        # 结尾提示
        answer_parts.append(f"\n（基于知识库中{len(contexts)}条相关文档生成）")
        
        return '\n'.join(answer_parts)
    
    def get_optimization_advice(self, context: Dict[str, Any]) -> str:
        """
        获取能耗优化建议
        
        Args:
            context: 建筑上下文信息，包含：
                - building_type: 建筑类型
                - current_temp: 当前温度
                - target_temp: 目标温度
                - occupancy: 人员数量
                - time_of_day: 时间
                - energy_consumption: 当前能耗
                
        Returns:
            优化建议文本
        """
        # 构建查询
        building_type = context.get('building_type', '建筑')
        current_temp = context.get('current_temp', 25)
        target_temp = context.get('target_temp', 24)
        occupancy = context.get('occupancy', 0)
        time_of_day = context.get('time_of_day', '白天')
        
        query_parts = [f"{building_type}能耗优化"]
        
        if current_temp > target_temp:
            query_parts.append("制冷优化")
        else:
            query_parts.append("制热优化")
        
        if occupancy > 0:
            query_parts.append("人员热负荷")
        
        query = " ".join(query_parts)
        
        # 查询知识库
        result = self.query(query, top_k=3)
        
        # 生成具体建议
        advice_parts = ["基于知识库的能耗优化建议：\n"]
        
        # 温度控制建议
        temp_diff = abs(current_temp - target_temp)
        if temp_diff > 2:
            advice_parts.append(
                f"1. 温度调节：当前温度{current_temp}°C与目标温度{target_temp}°C相差较大，"
                f"建议逐步调节，避免温度骤变增加能耗。"
            )
        
        # 人员相关建议
        if occupancy == 0:
            advice_parts.append(
                "2. 人员检测：当前无人 occupancy=0，建议降低HVAC运行强度或切换至节能模式。"
            )
        elif occupancy > 20:
            advice_parts.append(
                f"2. 人员负荷：当前人员较多({occupancy}人)，注意增加新风量，"
                f"同时可利用人员热负荷辅助采暖。"
            )
        
        # 添加知识库查询结果
        if result.confidence > 0.5:
            advice_parts.append(f"\n3. 参考方案：\n{result.answer}")
        
        return "\n".join(advice_parts)
    
    def _save_index(self) -> None:
        """保存索引到磁盘"""
        # 保存文档列表
        docs_data = {
            doc_id: {
                'content': doc.content,
                'metadata': doc.metadata
            }
            for doc_id, doc in self._documents.items()
        }
        
        with open(self._index_dir / "documents.json", 'w', encoding='utf-8') as f:
            json.dump(docs_data, f, ensure_ascii=False)
        
        # 保存向量存储
        if self.vector_store:
            self.vector_store.save(str(self._index_dir / "vectors.json"))
        
        # 保存知识图谱
        self.knowledge_graph.save(str(self._index_dir / "knowledge_graph.json"))
        
        # 保存状态
        state = {
            'indexed': self._indexed,
            'document_count': len(self._documents),
            'chunk_count': len(self._chunks),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self._index_dir / "state.json", 'w') as f:
            json.dump(state, f)
        
        logger.info(f"Index saved to {self._index_dir}")
    
    def _load_index(self) -> None:
        """从磁盘加载索引"""
        state_path = self._index_dir / "state.json"
        
        if not state_path.exists():
            logger.info("No existing index found")
            return
        
        try:
            # 加载状态
            with open(state_path, 'r') as f:
                state = json.load(f)
            
            self._indexed = state.get('indexed', False)
            
            # 加载文档
            docs_path = self._index_dir / "documents.json"
            if docs_path.exists():
                with open(docs_path, 'r', encoding='utf-8') as f:
                    docs_data = json.load(f)
                
                for doc_id, data in docs_data.items():
                    doc = Document(
                        content=data['content'],
                        metadata=data['metadata'],
                        doc_id=doc_id
                    )
                    self._documents[doc_id] = doc
            
            # 加载向量存储
            vectors_path = self._index_dir / "vectors.json"
            if vectors_path.exists():
                self.vector_store = VectorStore(
                    embedding_dim=self.embedding_provider.embedding_dim,
                    use_faiss=True
                )
                self.vector_store.load(str(vectors_path))
            
            # 加载知识图谱
            kg_path = self._index_dir / "knowledge_graph.json"
            if kg_path.exists():
                self.knowledge_graph.load(str(kg_path))
            
            logger.info(f"Index loaded from {self._index_dir}")
            logger.info(f"  - Documents: {len(self._documents)}")
            logger.info(f"  - Indexed: {self._indexed}")
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            self._indexed = False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        return {
            'document_count': len(self._documents),
            'chunk_count': len(self._chunks),
            'entity_count': len(self.knowledge_graph._entities),
            'relation_count': len(self.knowledge_graph._relations),
            'indexed': self._indexed,
            'root_dir': str(self.root_dir)
        }
    
    def clear(self) -> None:
        """清空知识库"""
        self._documents.clear()
        self._chunks.clear()
        self.knowledge_graph = KnowledgeGraph()
        self.vector_store = None
        self._indexed = False
        
        # 删除索引文件
        import shutil
        if self._index_dir.exists():
            shutil.rmtree(self._index_dir)
            self._index_dir.mkdir(exist_ok=True)
        
        logger.info("Knowledge base cleared")
