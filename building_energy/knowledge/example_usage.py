"""
知识库模块使用示例

演示如何使用KnowledgeBase进行文档管理和查询。
"""

from building_energy.knowledge import KnowledgeBase, DocumentLoader


def main():
    """主函数"""
    # 1. 创建知识库
    kb = KnowledgeBase("./knowledge_base")
    
    # 2. 添加文档
    # 添加单个文档
    kb.add_document("docs/hvac_guide.md")
    
    # 批量添加文档
    kb.add_documents([
        "docs/lighting_guide.md",
        "docs/energy_policy.pdf"
    ])
    
    # 3. 构建索引
    print("Building index...")
    kb.index()
    
    # 4. 查询知识库
    result = kb.query("如何优化空调能耗？", top_k=5)
    print(f"\nQuery: 如何优化空调能耗？")
    print(f"Answer: {result.answer}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Sources: {len(result.sources)}")
    
    # 5. 获取能耗优化建议
    context = {
        'building_type': '办公楼',
        'current_temp': 28,
        'target_temp': 24,
        'occupancy': 15,
        'time_of_day': '下午',
        'energy_consumption': 150.5
    }
    
    advice = kb.get_optimization_advice(context)
    print(f"\nOptimization Advice:")
    print(advice)
    
    # 6. 查看统计信息
    stats = kb.get_stats()
    print(f"\nKnowledge Base Stats:")
    print(f"  Documents: {stats['document_count']}")
    print(f"  Chunks: {stats['chunk_count']}")
    print(f"  Entities: {stats['entity_count']}")
    print(f"  Relations: {stats['relation_count']}")
    print(f"  Indexed: {stats['indexed']}")


if __name__ == "__main__":
    main()
