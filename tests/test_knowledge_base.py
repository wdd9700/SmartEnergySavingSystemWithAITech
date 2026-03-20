"""
知识库模块单元测试

测试内容：
- DocumentLoader: 文档加载和解析
- KnowledgeBase: 知识库核心功能
- 向量检索和查询
"""

import os
import sys
import tempfile
import shutil
import unittest
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from building_energy.knowledge.document_loader import (
    DocumentLoader, Document, DocumentChunk,
    MarkdownParser, TextParser
)
from building_energy.knowledge.graph_rag import (
    KnowledgeBase, QueryResult, Entity, Relation,
    EmbeddingProvider, VectorStore, KnowledgeGraph
)


class TestMarkdownParser(unittest.TestCase):
    """测试Markdown解析器"""
    
    def setUp(self):
        self.parser = MarkdownParser()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_supports_markdown(self):
        """测试支持Markdown文件"""
        self.assertTrue(self.parser.supports("test.md"))
        self.assertTrue(self.parser.supports("test.MD"))
        self.assertFalse(self.parser.supports("test.txt"))
        self.assertFalse(self.parser.supports("test.pdf"))
    
    def test_parse_markdown(self):
        """测试解析Markdown内容"""
        # 创建测试文件
        test_file = Path(self.temp_dir) / "test.md"
        content = """# 测试标题

这是第一段内容。

## 子标题

这是第二段内容，包含一些关键词。
"""
        test_file.write_text(content, encoding='utf-8')
        
        # 解析
        doc = self.parser.parse(str(test_file))
        
        # 验证
        self.assertEqual(doc.metadata['title'], '测试标题')
        self.assertEqual(doc.metadata['file_type'], 'markdown')
        self.assertIn('测试标题', doc.content)
        self.assertIn('子标题', doc.content)
    
    def test_parse_with_frontmatter(self):
        """测试解析带YAML frontmatter的Markdown"""
        test_file = Path(self.temp_dir) / "test_with_frontmatter.md"
        content = """---
title: 文档标题
author: 测试作者
date: 2024-01-01
---

# 正文标题

正文内容。
"""
        test_file.write_text(content, encoding='utf-8')
        
        doc = self.parser.parse(str(test_file))
        
        # 验证frontmatter被解析
        self.assertEqual(doc.metadata.get('author'), '测试作者')
        self.assertEqual(doc.metadata.get('date'), '2024-01-01')


class TestTextParser(unittest.TestCase):
    """测试文本解析器"""
    
    def setUp(self):
        self.parser = TextParser()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_parse_text(self):
        """测试解析文本文件"""
        test_file = Path(self.temp_dir) / "test.txt"
        content = "这是文本文件的第一行标题\n\n这是正文内容。"
        test_file.write_text(content, encoding='utf-8')
        
        doc = self.parser.parse(str(test_file))
        
        self.assertEqual(doc.metadata['file_type'], 'text')
        self.assertEqual(doc.metadata['title'], '这是文本文件的第一行标题')
        self.assertIn('正文内容', doc.content)


class TestDocumentLoader(unittest.TestCase):
    """测试文档加载器"""
    
    def setUp(self):
        self.loader = DocumentLoader(chunk_size=100, chunk_overlap=20)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_load_single_file(self):
        """测试加载单个文件"""
        test_file = Path(self.temp_dir) / "test.md"
        test_file.write_text("# 标题\n\n内容", encoding='utf-8')
        
        doc = self.loader.load(str(test_file))
        
        self.assertIsInstance(doc, Document)
        self.assertIsNotNone(doc.doc_id)
        self.assertIn('标题', doc.content)
    
    def test_load_directory(self):
        """测试加载目录"""
        # 创建多个文件
        (Path(self.temp_dir) / "doc1.md").write_text("# Doc1", encoding='utf-8')
        (Path(self.temp_dir) / "doc2.md").write_text("# Doc2", encoding='utf-8')
        (Path(self.temp_dir) / "doc3.txt").write_text("Doc3", encoding='utf-8')
        
        docs = self.loader.load_directory(self.temp_dir)
        
        self.assertEqual(len(docs), 3)
    
    def test_chunk_document(self):
        """测试文档分块"""
        doc = Document(
            content="第一段内容。\n\n第二段内容。\n\n第三段内容。",
            metadata={'title': 'Test'}
        )
        
        chunks = self.loader.chunk_document(doc)
        
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        
        # 验证块结构
        for chunk in chunks:
            self.assertIsInstance(chunk, DocumentChunk)
            self.assertEqual(chunk.doc_id, doc.doc_id)
            self.assertIsNotNone(chunk.chunk_id)
    
    def test_file_not_found(self):
        """测试文件不存在时抛出异常"""
        with self.assertRaises(FileNotFoundError):
            self.loader.load("/nonexistent/file.md")
    
    def test_unsupported_file_type(self):
        """测试不支持的文件类型"""
        test_file = Path(self.temp_dir) / "test.xyz"
        test_file.write_text("content", encoding='utf-8')
        
        with self.assertRaises(ValueError):
            self.loader.load(str(test_file))


class TestEmbeddingProvider(unittest.TestCase):
    """测试向量嵌入提供器"""
    
    def setUp(self):
        self.provider = EmbeddingProvider()
    
    def test_embed_single_text(self):
        """测试单文本嵌入"""
        texts = ["这是一个测试句子"]
        embeddings = self.provider.embed(texts)
        
        self.assertEqual(embeddings.shape[0], 1)
        self.assertEqual(embeddings.shape[1], self.provider.embedding_dim)
    
    def test_embed_multiple_texts(self):
        """测试多文本嵌入"""
        texts = ["文本一", "文本二", "文本三"]
        embeddings = self.provider.embed(texts)
        
        self.assertEqual(embeddings.shape[0], 3)
        self.assertEqual(embeddings.shape[1], self.provider.embedding_dim)
    
    def test_embedding_normalization(self):
        """测试向量归一化"""
        texts = ["测试文本"]
        embeddings = self.provider.embed(texts)
        
        # 向量应该大致归一化（长度接近1）
        norm = np.linalg.norm(embeddings[0])
        self.assertAlmostEqual(norm, 1.0, delta=0.1)


class TestVectorStore(unittest.TestCase):
    """测试向量存储"""
    
    def setUp(self):
        self.store = VectorStore(embedding_dim=384, use_faiss=False)
        
        # 添加测试数据
        vectors = np.random.randn(5, 384).astype('float32')
        # 归一化
        for i in range(5):
            vectors[i] = vectors[i] / np.linalg.norm(vectors[i])
        
        metadata = [
            {'id': i, 'text': f'doc_{i}'}
            for i in range(5)
        ]
        
        self.store.add(vectors, metadata)
    
    def test_search_returns_results(self):
        """测试搜索返回结果"""
        query = np.random.randn(384).astype('float32')
        query = query / np.linalg.norm(query)
        
        results = self.store.search(query, top_k=3)
        
        self.assertEqual(len(results), 3)
        
        # 验证结果格式
        for meta, score in results:
            self.assertIn('id', meta)
            self.assertIn('text', meta)
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, -1.0)
            self.assertLessEqual(score, 1.0)
    
    def test_search_sorted_by_similarity(self):
        """测试结果按相似度排序"""
        query = np.random.randn(384).astype('float32')
        query = query / np.linalg.norm(query)
        
        results = self.store.search(query, top_k=5)
        
        # 验证按相似度降序排列
        scores = [score for _, score in results]
        self.assertEqual(scores, sorted(scores, reverse=True))


class TestKnowledgeGraph(unittest.TestCase):
    """测试知识图谱"""
    
    def setUp(self):
        self.graph = KnowledgeGraph()
    
    def test_add_entity(self):
        """测试添加实体"""
        entity = Entity(
            name="空调系统",
            entity_type="HVAC_SYSTEM",
            properties={'description': '中央空调'}
        )
        
        self.graph.add_entity(entity)
        
        retrieved = self.graph.get_entity("空调系统")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.entity_type, "HVAC_SYSTEM")
    
    def test_add_relation(self):
        """测试添加关系"""
        # 添加实体
        entity1 = Entity(name="空调", entity_type="HVAC")
        entity2 = Entity(name="温度传感器", entity_type="SENSOR")
        
        self.graph.add_entity(entity1)
        self.graph.add_entity(entity2)
        
        # 添加关系
        relation = Relation(
            source="空调",
            target="温度传感器",
            relation_type="has_sensor"
        )
        
        self.graph.add_relation(relation)
        
        # 验证关系
        related = self.graph.get_related_entities("空调")
        self.assertEqual(len(related), 1)
        self.assertEqual(related[0][0].name, "温度传感器")
    
    def test_extract_from_text(self):
        """测试从文本提取实体"""
        text = "空调系统使用PID控制策略来调节温度。"
        
        entities, relations = self.graph.extract_from_text(text)
        
        self.assertIsInstance(entities, list)
        self.assertIsInstance(relations, list)
        
        # 应该提取到一些实体
        self.assertGreater(len(entities), 0)


class TestKnowledgeBase(unittest.TestCase):
    """测试知识库核心功能"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.kb_dir = Path(self.temp_dir) / "knowledge_base"
        self.kb = KnowledgeBase(str(self.kb_dir))
        
        # 创建测试文档
        self.docs_dir = Path(self.temp_dir) / "docs"
        self.docs_dir.mkdir()
        
        # HVAC文档
        hvac_doc = self.docs_dir / "hvac_guide.md"
        hvac_doc.write_text("""# HVAC系统优化指南

## 温度控制策略

空调系统的温度控制是能耗管理的关键。
建议将夏季设定温度设为26°C，冬季设为20°C。

## PID控制参数调优

PID控制器需要合理设置参数：
- P（比例）：控制响应速度
- I（积分）：消除稳态误差
- D（微分）：抑制超调

## 节能建议

1. 定期清洗过滤网
2. 检查制冷剂压力
3. 优化送风温度
""", encoding='utf-8')
        
        # 照明文档
        lighting_doc = self.docs_dir / "lighting.md"
        lighting_doc.write_text("""# 智能照明控制

## 人因照明

根据时间调节色温可以提高舒适度：
- 早晨：高色温（5000K）
- 下午：中色温（4000K）
- 晚上：低色温（2700K）

## 自然光利用

结合光照传感器和窗帘控制，最大化利用自然光。
""", encoding='utf-8')
        
        self.hvac_doc_path = str(hvac_doc)
        self.lighting_doc_path = str(lighting_doc)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """测试知识库初始化"""
        self.assertIsNotNone(self.kb.document_loader)
        self.assertIsNotNone(self.kb.embedding_provider)
        self.assertIsNotNone(self.kb.knowledge_graph)
        self.assertFalse(self.kb._indexed)
    
    def test_add_document(self):
        """测试添加文档"""
        self.kb.add_document(self.hvac_doc_path)
        
        self.assertEqual(len(self.kb._documents), 1)
        self.assertFalse(self.kb._indexed)  # 添加后应标记为未索引
    
    def test_add_multiple_documents(self):
        """测试批量添加文档"""
        self.kb.add_documents([self.hvac_doc_path, self.lighting_doc_path])
        
        self.assertEqual(len(self.kb._documents), 2)
    
    def test_index(self):
        """测试构建索引"""
        self.kb.add_document(self.hvac_doc_path)
        self.kb.index()
        
        self.assertTrue(self.kb._indexed)
        self.assertIsNotNone(self.kb.vector_store)
        self.assertGreater(len(self.kb._chunks), 0)
    
    def test_query_before_index(self):
        """测试索引前查询应返回提示"""
        result = self.kb.query("如何优化空调？")
        
        self.assertIn("尚未索引", result.answer)
        self.assertEqual(result.confidence, 0.0)
    
    def test_query_after_index(self):
        """测试索引后查询"""
        self.kb.add_document(self.hvac_doc_path)
        self.kb.index()
        
        result = self.kb.query("空调温度控制")
        
        self.assertIsInstance(result, QueryResult)
        self.assertIsNotNone(result.answer)
        self.assertGreater(len(result.sources), 0)
        self.assertGreater(result.confidence, 0)
    
    def test_query_sources(self):
        """测试查询返回的来源信息"""
        self.kb.add_document(self.hvac_doc_path)
        self.kb.index()
        
        result = self.kb.query("PID控制", top_k=3)
        
        self.assertLessEqual(len(result.sources), 3)
        
        for source in result.sources:
            self.assertIn('doc_id', source)
            self.assertIn('chunk_id', source)
            self.assertIn('similarity', source)
    
    def test_generate_answer_with_contexts(self):
        """测试_generate_answer方法 - 有上下文"""
        self.kb.add_document(self.hvac_doc_path)
        self.kb.index()
        
        # 测试生成答案
        contexts = [
            "建议将夏季空调设定温度设为26°C，冬季设为20°C。定期清洗过滤网可以提高效率。",
            "PID控制器需要合理设置参数：P控制响应速度，I消除稳态误差，D抑制超调。",
            "空调系统能耗占建筑总能耗的40-60%，优化控制策略可以显著节能。"
        ]
        
        answer = self.kb._generate_answer("如何优化空调能耗？", contexts)
        
        # 验证答案结构
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 50)
        # 应该包含要点编号
        self.assertIn("1.", answer)
        # 应该包含问题类型识别
        self.assertIn("空调", answer)
    
    def test_generate_answer_empty_contexts(self):
        """测试_generate_answer方法 - 空上下文"""
        answer = self.kb._generate_answer("测试问题", [])
        
        self.assertEqual(answer, "抱歉，知识库中没有找到相关信息。")
    
    def test_generate_answer_temperature_question(self):
        """测试_generate_answer方法 - 温度相关问题"""
        contexts = [
            "夏季空调设定温度建议为26°C，每提高1°C可节能约6-8%。",
            "冬季采暖设定温度建议为20°C，过高的设定温度会增加能耗。",
            "温度传感器应安装在代表性位置，避免阳光直射和热源影响。"
        ]
        
        answer = self.kb._generate_answer("空调温度应该如何设置？", contexts)
        
        # 验证答案包含温度相关信息
        self.assertIn("温度", answer)
        self.assertIn("26", answer)
        self.assertIn("建议", answer)
        self.assertIn("1.", answer)
    
    def test_get_optimization_advice(self):
        """测试获取优化建议"""
        self.kb.add_document(self.hvac_doc_path)
        self.kb.index()
        
        context = {
            'building_type': '办公楼',
            'current_temp': 28,
            'target_temp': 24,
            'occupancy': 10,
            'time_of_day': '下午'
        }
        
        advice = self.kb.get_optimization_advice(context)
        
        self.assertIsInstance(advice, str)
        self.assertIn("能耗优化", advice)
        self.assertIn("温度", advice)
    
    def test_get_stats(self):
        """测试获取统计信息"""
        self.kb.add_document(self.hvac_doc_path)
        
        stats = self.kb.get_stats()
        
        self.assertEqual(stats['document_count'], 1)
        self.assertIn('chunk_count', stats)
        self.assertIn('entity_count', stats)
        self.assertIn('indexed', stats)
    
    def test_clear(self):
        """测试清空知识库"""
        self.kb.add_document(self.hvac_doc_path)
        self.kb.index()
        
        self.kb.clear()
        
        self.assertEqual(len(self.kb._documents), 0)
        self.assertEqual(len(self.kb._chunks), 0)
        self.assertFalse(self.kb._indexed)
    
    def test_save_and_load_index(self):
        """测试索引保存和加载"""
        # 创建并索引
        self.kb.add_document(self.hvac_doc_path)
        self.kb.index()
        
        # 创建新的知识库实例（应该加载已有索引）
        kb2 = KnowledgeBase(str(self.kb_dir))
        
        self.assertEqual(len(kb2._documents), 1)
        self.assertTrue(kb2._indexed)


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.kb_dir = Path(self.temp_dir) / "kb"
        self.docs_dir = Path(self.temp_dir) / "docs"
        self.docs_dir.mkdir()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_full_workflow(self):
        """测试完整工作流程"""
        # 1. 创建测试文档
        doc_path = self.docs_dir / "energy_saving.md"
        doc_path.write_text("""# 建筑节能最佳实践

## HVAC优化

空调系统能耗占建筑总能耗的40-60%。
优化策略包括：
- 定期维护
- 智能控制
- 负荷预测

## 照明优化

LED照明比传统灯具节能50-70%。
结合自然光传感器可实现自动调光。
""", encoding='utf-8')
        
        # 2. 创建知识库
        kb = KnowledgeBase(str(self.kb_dir))
        
        # 3. 添加文档
        kb.add_document(str(doc_path))
        self.assertEqual(len(kb._documents), 1)
        
        # 4. 构建索引
        kb.index()
        self.assertTrue(kb._indexed)
        
        # 5. 查询
        result = kb.query("空调能耗优化")
        self.assertGreater(result.confidence, 0)
        self.assertGreater(len(result.sources), 0)
        
        # 6. 获取优化建议
        advice = kb.get_optimization_advice({
            'building_type': '教学楼',
            'current_temp': 27,
            'target_temp': 25,
            'occupancy': 30
        })
        self.assertIn("能耗优化", advice)
        
        # 7. 验证统计
        stats = kb.get_stats()
        self.assertGreater(stats['document_count'], 0)


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestMarkdownParser))
    suite.addTests(loader.loadTestsFromTestCase(TestTextParser))
    suite.addTests(loader.loadTestsFromTestCase(TestDocumentLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestEmbeddingProvider))
    suite.addTests(loader.loadTestsFromTestCase(TestVectorStore))
    suite.addTests(loader.loadTestsFromTestCase(TestKnowledgeGraph))
    suite.addTests(loader.loadTestsFromTestCase(TestKnowledgeBase))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    import numpy as np
    success = run_tests()
    sys.exit(0 if success else 1)
