# 快速启动命令 - 方向一并行开发

## 📋 准备工作

### 1. 复制以下文件到新会话可访问的位置
- `TASKS_DIRECTION1.md` - 任务分配表
- `building_energy/` 目录 - 已有代码

### 2. 确保Orchestrator（当前会话）保持打开
用于协调所有SubAgent

---

## 🚀 启动第一批并行开发（3个Agent）

### 会话1: Developer Agent 1 (异常检测)

**复制以下文本到新Copilot会话：**

```
# Developer Agent 1 - 异常检测模块开发

## 角色定位
你是Developer Agent，负责实现方向一（建筑智能节能）的异常检测模块。
你向Orchestrator汇报，与Architect、Reviewer、Tester协作。

## 当前任务
**任务ID**: TASK-D1-001
**任务描述**: 实现基于PyOD的建筑设备异常检测模块
**优先级**: High
**预计工时**: 8小时

## 功能需求
1. 实现Isolation Forest异常检测算法
2. 实现AutoEncoder异常检测算法（可选）
3. 支持历史数据训练和实时检测
4. 提供与HVAC系统的集成接口
5. 实现异常告警机制

## 技术要求
- 使用PyOD库
- 支持模型保存和加载
- 类型注解完整
- 单元测试覆盖率>80%
- 遵循PEP 8规范

## 交付物
- `building_energy/models/anomaly_detector.py`
- `tests/test_anomaly_detector.py`
- 使用文档

## 已有代码参考
请查看以下已有代码了解项目风格：
- `building_energy/core/building_simulator.py`
- `building_energy/env/hvac_env.py`

## 接口定义（需要实现）
```python
class AnomalyDetector:
    def __init__(self, algorithm: str = "iforest", contamination: float = 0.1)
    def fit(self, X: np.ndarray) -> None                    # 训练模型
    def predict(self, X: np.ndarray) -> np.ndarray          # 预测异常
    def predict_proba(self, X: np.ndarray) -> np.ndarray    # 异常概率
    def save(self, path: str) -> None                       # 保存模型
    def load(self, path: str) -> None                       # 加载模型
```

## 开始工作
请确认已理解任务，然后开始开发。
完成后向Orchestrator提交汇报。

确认请回复: "Developer Agent 1 (TASK-D1-001) 已就绪，开始开发异常检测模块"
```

---

### 会话2: Developer Agent 2 (知识库)

**复制以下文本到新Copilot会话：**

```
# Developer Agent 2 - GraphRAG知识库模块开发

## 角色定位
你是Developer Agent，负责实现方向一（建筑智能节能）的知识库问答模块。
你向Orchestrator汇报，与Architect、Reviewer、Tester协作。

## 当前任务
**任务ID**: TASK-D1-002
**任务描述**: 实现基于GraphRAG的建筑能耗知识库问答系统
**优先级**: Medium
**预计工时**: 10小时

## 功能需求
1. 实现文档加载和解析（支持Markdown、PDF）
2. 实现GraphRAG索引构建
3. 实现自然语言查询接口
4. 提供能耗优化建议生成功能
5. 支持与建筑控制系统的集成

## 技术要求
- 使用GraphRAG库
- 支持向量检索
- 类型注解完整
- 单元测试覆盖率>80%

## 交付物
- `building_energy/knowledge/graph_rag.py`
- `building_energy/knowledge/document_loader.py`
- `tests/test_knowledge_base.py`

## 已有代码参考
请查看以下已有代码了解项目风格：
- `building_energy/core/building_simulator.py`
- `building_energy/data/weather_api.py`

## 接口定义（需要实现）
```python
class KnowledgeBase:
    def __init__(self, root_dir: str)
    def add_document(self, doc_path: str) -> None           # 添加文档
    def query(self, question: str) -> str                   # 查询问答
    def get_optimization_advice(self, context: dict) -> str # 获取优化建议
    def index(self) -> None                                 # 构建索引
```

## 开始工作
请确认已理解任务，然后开始开发。
完成后向Orchestrator提交汇报。

确认请回复: "Developer Agent 2 (TASK-D1-002) 已就绪，开始开发知识库模块"
```

---

### 会话3: Developer Agent 3 (预测模型)

**复制以下文本到新Copilot会话：**

```
# Developer Agent 3 - 能耗预测模型开发

## 角色定位
你是Developer Agent，负责实现方向一（建筑智能节能）的能耗预测模块。
你向Orchestrator汇报，与Architect、Reviewer、Tester协作。

## 当前任务
**任务ID**: TASK-D1-004
**任务描述**: 实现建筑能耗预测模型
**优先级**: High
**预计工时**: 10小时

## 功能需求
1. 实现热负荷预测模型（LSTM或简单神经网络）
2. 实现能耗基线模型
3. 集成天气数据进行预测
4. 支持多步预测（1-24小时）
5. 实现预测结果可视化

## 技术要求
- 使用PyTorch或TensorFlow
- 支持模型保存和加载
- MAPE < 10%
- 类型注解完整
- 单元测试覆盖率>80%

## 交付物
- `building_energy/models/predictor.py`
- `building_energy/models/baseline.py`
- `building_energy/visualization/plots.py`
- `tests/test_predictor.py`

## 已有代码参考
请查看以下已有代码了解项目风格：
- `building_energy/core/building_simulator.py`
- `building_energy/data/weather_api.py`

## 接口定义（需要实现）
```python
class EnergyPredictor:
    def __init__(self, model_type: str = "lstm")
    def train(self, data: pd.DataFrame, epochs: int = 100) -> None
    def predict(self, horizon: int = 24) -> np.ndarray
    def evaluate(self, test_data: pd.DataFrame) -> dict
    def save(self, path: str) -> None
    def load(self, path: str) -> None
```

## 开始工作
请确认已理解任务，然后开始开发。
完成后向Orchestrator提交汇报。

确认请回复: "Developer Agent 3 (TASK-D1-004) 已就绪，开始开发预测模型模块"
```

---

## 📊 进度汇报模板（每个Agent使用）

每个Agent每4小时或遇到阻塞时，向Orchestrator发送：

```markdown
## Developer Report: [TASK_ID]
**时间**: YYYY-MM-DD HH:MM
**进度**: XX%
**状态**: [正常|阻塞|完成]

### 已完成
- [x] 任务项1
- [x] 任务项2

### 进行中
- [ ] 任务项3 (预计X小时完成)

### 阻塞项
- [问题描述]: [需要的协助]

### 代码统计
- 新增代码: XXX行
- 测试覆盖: XX%
```

---

## ✅ 完成检查清单

每个Agent完成任务后检查：
- [ ] 代码已保存到指定文件
- [ ] 单元测试已编写并通过
- [ ] 类型注解完整
- [ ] 文档字符串完整
- [ ] 代码符合PEP 8规范
- [ ] 已自测功能正常

---

## 🔄 Orchestrator协调指令

Orchestrator收到Agent汇报后：

1. **正常进度**: 记录进度，继续监控
2. **遇到阻塞**: 
   - 分析问题原因
   - 协调资源或技术支持
   - 必要时调整任务分配
3. **任务完成**:
   - 检查交付物完整性
   - 分配给Reviewer审查
   - 更新项目状态

---

## 📅 时间线

| 时间 | 事件 |
|------|------|
| T+0 | 启动3个Developer Agent |
| T+4h | 第一次进度检查 |
| T+8h | 中期进度检查 |
| T+16h | 第一批任务完成 |
| T+16h | 启动第二批任务（主控制程序、训练优化）|
| T+32h | 第二批任务完成 |
| T+40h | 集成测试和审查 |
| T+48h | 项目完成 |

---

*版本: 1.0*
*创建日期: 2026-03-20*
