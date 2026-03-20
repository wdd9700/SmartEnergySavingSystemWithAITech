# 建筑智能节能系统 - 架构图

本文档使用 Mermaid 语法绘制系统的各种架构图。

---

## 1. 系统整体架构图

```mermaid
graph TB
    subgraph "数据输入层"
        A1[传感器数据<br/>温度/湿度/CO2]
        A2[天气API<br/>外部气象数据]
        A3[摄像头<br/>人员检测]
        A4[建筑模型<br/>EnergyPlus]
    end

    subgraph "核心处理层"
        B1[异常检测模块<br/>PyOD]
        B2[预测模型<br/>LSTM/Transformer]
        B3[知识库<br/>GraphRAG]
        B4[强化学习<br/>SB3]
    end

    subgraph "控制决策层"
        C1[HVAC控制<br/>温度/风量]
        C2[照明控制<br/>开关/亮度]
        C3[告警管理<br/>分级通知]
    end

    subgraph "输出层"
        D1[Web仪表盘]
        D2[CLI接口]
        D3[数据存储]
        D4[第三方系统]
    end

    A1 --> B1
    A1 --> B2
    A2 --> B2
    A3 --> B2
    A4 --> B4

    B1 --> C3
    B2 --> C1
    B2 --> C2
    B3 --> C1
    B4 --> C1

    C1 --> D1
    C1 --> D3
    C2 --> D1
    C3 --> D1
    C3 --> D4

    D1 --> D2
```

---

## 2. 数据流图

```mermaid
flowchart LR
    subgraph "数据采集"
        S1[传感器]
        S2[摄像头]
        S3[天气API]
        S4[建筑模拟]
    end

    subgraph "数据预处理"
        P1[数据清洗]
        P2[特征工程]
        P3[标准化]
    end

    subgraph "模型推理"
        M1[异常检测]
        M2[能耗预测]
        M3[RL策略]
    end

    subgraph "决策执行"
        E1[控制指令]
        E2[告警通知]
        E3[数据存储]
    end

    S1 --> P1
    S2 --> P1
    S3 --> P2
    S4 --> P2

    P1 --> P3
    P2 --> P3

    P3 --> M1
    P3 --> M2
    P3 --> M3

    M1 --> E2
    M2 --> E1
    M3 --> E1

    E1 --> E3
    E2 --> E3
```

---

## 3. 模块依赖图

```mermaid
graph TD
    subgraph "配置层"
        CFG[ConfigManager<br/>配置管理]
    end

    subgraph "数据层"
        DATA[DataRecorder<br/>数据记录]
        WEATHER[WeatherAPI<br/>天气接口]
    end

    subgraph "模型层"
        ANOMALY[AnomalyDetector<br/>异常检测]
        PREDICT[Predictor<br/>能耗预测]
        BASELINE[Baseline<br/>基线模型]
    end

    subgraph "知识层"
        KB[KnowledgeBase<br/>知识库]
        DOC[DocumentLoader<br/>文档加载]
        GRAPH[GraphRAG<br/>图检索]
    end

    subgraph "控制层"
        CTRL[BuildingController<br/>主控制器]
        CLI[CLI<br/>命令行接口]
    end

    subgraph "环境层"
        ENV[HVACEnv<br/>RL环境]
        SIM[BuildingSimulator<br/>建筑模拟]
    end

    CFG --> CTRL
    CFG --> ENV

    DATA --> ANOMALY
    DATA --> PREDICT
    WEATHER --> PREDICT

    DOC --> KB
    GRAPH --> KB

    ANOMALY --> CTRL
    PREDICT --> CTRL
    KB --> CTRL

    CTRL --> CLI

    ENV --> CTRL
    SIM --> ENV
```

---

## 4. 部署架构图

```mermaid
graph TB
    subgraph "边缘层 Edge"
        E1[摄像头<br/>人员检测]
        E2[传感器节点<br/>温度/湿度]
        E3[照明控制器<br/>楼道灯控]
        E4[空调控制器<br/>教室AC]
    end

    subgraph "网关层 Gateway"
        G1[边缘网关<br/>Jetson Nano]
        G2[数据聚合]
        G3[本地推理]
    end

    subgraph "云端层 Cloud"
        C1[应用服务器<br/>API服务]
        C2[模型训练<br/>GPU集群]
        C3[知识库<br/>向量数据库]
        C4[时序数据库<br/>InfluxDB]
    end

    subgraph "用户层 User"
        U1[Web仪表盘]
        U2[移动App]
        U3[管理CLI]
    end

    E1 --> G1
    E2 --> G1
    E3 --> G1
    E4 --> G1

    G1 --> G2
    G1 --> G3

    G2 --> C4
    G3 --> C1

    C1 --> C3
    C2 --> C1
    C4 --> C2

    C1 --> U1
    C1 --> U2
    C1 --> U3
```

---

## 5. 强化学习训练流程图

```mermaid
sequenceDiagram
    participant User as 用户
    participant Trainer as 训练脚本
    participant Env as HVAC环境
    participant SB3 as Stable-Baselines3
    participant Model as 策略模型

    User->>Trainer: 启动训练
    Trainer->>Env: 创建环境
    Env->>Env: 初始化建筑模拟
    
    loop 训练循环
        Trainer->>SB3: 创建算法(SAC/PPO)
        SB3->>Model: 初始化策略网络
        
        loop 每个时间步
            SB3->>Model: 预测动作
            Model-->>SB3: 动作值
            SB3->>Env: 执行动作
            Env->>Env: 模拟建筑响应
            Env-->>SB3: 返回状态/奖励
            SB3->>SB3: 存储经验
            SB3->>Model: 更新策略
        end
        
        SB3-->>Trainer: 返回训练结果
    end
    
    Trainer->>Trainer: 保存模型
    Trainer-->>User: 训练完成
```

---

## 6. 知识库查询流程图

```mermaid
sequenceDiagram
    participant User as 用户
    participant CLI as CLI接口
    participant KB as KnowledgeBase
    participant Emb as Embedding模型
    participant FAISS as 向量数据库
    participant LLM as 语言模型

    User->>CLI: 输入查询
    CLI->>KB: query(question)
    
    KB->>Emb: 编码查询
    Emb-->>KB: 查询向量
    
    KB->>FAISS: 相似度检索
    FAISS-->>KB: 相关文档
    
    KB->>KB: 构建上下文
    KB->>LLM: 生成回答
    LLM-->>KB: 自然语言回答
    
    KB-->>CLI: 返回结果
    CLI-->>User: 显示答案
```

---

## 7. 异常检测流程图

```mermaid
flowchart TD
    A[原始数据输入] --> B{数据类型}
    
    B -->|数值型| C[特征提取]
    B -->|时间序列| D[滑动窗口]
    
    C --> E[标准化处理]
    D --> E
    
    E --> F{检测模式}
    
    F -->|离线训练| G[模型训练<br/>Isolation Forest/LOF]
    F -->|在线检测| H[实时推理]
    
    G --> I[模型保存]
    I --> H
    
    H --> J{异常判断}
    J -->|异常| K[生成告警]
    J -->|正常| L[继续监控]
    
    K --> M[告警分级]
    M --> N[通知用户]
    
    L --> A
```

---

## 8. 系统启动时序图

```mermaid
sequenceDiagram
    participant User as 用户
    participant CLI as CLI
    participant Main as BuildingController
    participant Config as ConfigManager
    participant Modules as 各模块

    User->>CLI: beems start
    CLI->>Main: create_controller()
    
    Main->>Config: 加载配置
    Config-->>Main: 配置对象
    
    Main->>Main: 初始化日志
    
    par 并行初始化
        Main->>Modules: 初始化异常检测
        Main->>Modules: 初始化预测模型
        Main->>Modules: 初始化知识库
    end
    
    Modules-->>Main: 初始化完成
    
    Main->>Main: 启动监控线程
    Main->>Main: 启动控制循环
    
    Main-->>CLI: 系统就绪
    CLI-->>User: 启动成功
```

---

## 9. 技术栈架构图

```mermaid
graph TB
    subgraph "应用层"
        A1[CLI工具]
        A2[Web仪表盘]
        A3[API服务]
    end

    subgraph "业务逻辑层"
        B1[异常检测服务]
        B2[预测服务]
        B3[知识库服务]
        B4[控制服务]
    end

    subgraph "算法层"
        C1[PyOD<br/>异常检测]
        C2[PyTorch<br/>深度学习]
        C3[SB3<br/>强化学习]
        C4[FAISS<br/>向量检索]
    end

    subgraph "数据层"
        D1[NumPy/Pandas]
        D2[YAML/JSON]
        D3[SQLite/InfluxDB]
    end

    subgraph "基础设施层"
        E1[Python 3.10+]
        E2[OpenCV]
        E3[ONNX Runtime]
    end

    A1 --> B1
    A1 --> B2
    A1 --> B3
    A2 --> B4

    B1 --> C1
    B2 --> C2
    B3 --> C4
    B4 --> C3

    C1 --> D1
    C2 --> D1
    C3 --> D1
    C4 --> D3

    D1 --> E1
    D2 --> E1
    D3 --> E1
```

---

## 10. 数据模型关系图

```mermaid
erDiagram
    SENSOR_DATA {
        datetime timestamp
        string sensor_id
        float temperature
        float humidity
        float co2
        float power
    }
    
    ANOMALY_ALERT {
        string alert_id
        datetime timestamp
        string sensor_id
        string anomaly_type
        float severity
        string description
        bool resolved
    }
    
    KNOWLEDGE_DOC {
        string doc_id
        string title
        string content
        string source
        datetime created_at
        vector embedding
    }
    
    PREDICTION {
        string prediction_id
        datetime timestamp
        string model_id
        float predicted_value
        float confidence
        json features
    }
    
    CONTROL_ACTION {
        string action_id
        datetime timestamp
        string action_type
        string target_device
        json parameters
        string triggered_by
    }
    
    SENSOR_DATA ||--o{ ANOMALY_ALERT : generates
    SENSOR_DATA ||--o{ PREDICTION : inputs
    KNOWLEDGE_DOC ||--o{ CONTROL_ACTION : informs
```

---

## 图例说明

| 符号 | 含义 |
|------|------|
| ⭕ 圆形 | 开始/结束节点 |
| ▭ 矩形 | 处理步骤 |
| ◇ 菱形 | 判断/决策 |
| → 箭头 | 数据/控制流向 |
| -- 虚线 | 异步/可选流程 |

---

## 相关文档

- [方向一技术栈研究](../direction1_tech_stack_research.md)
- [方向二技术栈研究](../direction2_tech_stack_research.md)
- [方向三技术栈研究](../direction3_tech_stack_research.md)
- [项目需求文档](../project_requirements.md)
