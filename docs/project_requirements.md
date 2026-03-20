# 智能节能系统 - 项目需求文档

## 项目概述
基于AI的校园/建筑智能节能控制系统，包含建筑智能节能、交通节能、计算机节能三个主要方向。

---

## 方向一：建筑智能节能系统（优先级：最高）

### 1.1 天气预报集成与热负荷预测
**功能描述：**
- 获取实时天气数据（温度、湿度、风速、太阳辐射）
- 结合室内温度传感器数据
- 基于建筑热力学模型预测未来热负荷
- 动态调整空调运行策略

**技术约束：**
- 使用 OpenWeatherMap API 或 WeatherAPI.com
- 中国天气网API作为备选
- 免费额度内使用

### 1.2 数字孪生温度调控系统
**功能描述：**
- 输入：房间几何参数（墙厚、面积、朝向）、空调出风口位置/风速
- 输出：温度分布预测、最优控制策略
- 基于物理信息神经网络（PINN）或代理模型

**技术约束：**
- 使用 EnergyPlus + Python API 进行建筑能耗模拟
- CFD模拟数据训练神经网络（避免实时CFD计算开销）
- 参考：google/sbsim（Google的强化学习建筑模拟器）

### 1.3 建筑储能智能调度（电池储能）
**功能描述：**
- 根据电价策略（峰谷电价）优化储能充放电
- 根据电网电压/频率信号参与需求响应
- 预测性储能调度（结合天气预报）

**技术约束：**
- 储能类型：电池储能（如特斯拉Powerwall类似系统）
- 强化学习（PPO/SAC）优化充放电策略
- 预留与电网API对接接口

### 1.4 智能照明改进
**功能描述：**
- 人因照明（Human-centric Lighting）：根据时间调节色温
- 预测性开关：基于人员移动轨迹预测
- 自然光自适应：结合光照传感器和窗帘控制

### 1.5 设备故障检测（基于PID+数字孪生）
**功能描述：**
- 建立空调/风扇的正常运行数字孪生模型
- 实时监测参数偏差（温度下降速率、风速、功率）
- 使用PID控制器的残差分析
- 异常检测算法（Isolation Forest/Autoencoder）

**技术约束：**
- 复用现有传感器（温度、功率、风速）
- 无需额外硬件

---

## 方向二：交通节能系统

### 2.1 车辆检测与跟踪
**功能描述：**
- YOLOv8 车辆检测
- 多目标跟踪（DeepSORT/ByteTrack/BoT-SORT）
- 车辆类型分类（轿车/SUV/卡车/公交车/电动车/燃油车）

**技术约束：**
- 使用 Python 自动化脚本处理数据
- 使用 Ultralytics YOLO 内置跟踪功能

### 2.2 跨摄像头车辆匹配
**功能描述：**
- 车辆特征提取（外观、颜色、车型）
- 跨摄像头轨迹关联
- 路径-时间图生成

**技术约束：**
- 车辆Re-ID模型（OSNet、FastReID）
- 特征匹配算法（余弦相似度）
- 时空约束

### 2.3 交通信号优化
**功能描述：**
- 实时车流监测与预测
- 动态红绿灯时长调节
- 优先放行燃油车（减少怠速排放）

**技术约束：**
- 强化学习（RL）控制信号灯
- 与高德地图/交管部门数据对接（预留接口）
- 路网拓扑建模（图神经网络）

### 2.4 智能充电桩管理
**功能描述：**
- 基于车主日程规划的充电调度
- 动态监测电网电压/频率
- 错峰充电策略

---

## 方向三：计算机节能系统（实验室场景）

### 3.1 后台任务监控
**功能描述：**
- 定时获取进程列表（Windows/Linux）
- LLM辅助判断进程重要性
- 白名单机制（系统组件）
- 资源占用分析（CPU/GPU/内存）

**技术约束：**
- 约10-20台电脑
- 绝大部分为 Windows 11
- 少量 Linux 系统

### 3.2 智能关机决策
**功能描述：**
- 判断是否有长时间负载任务
- 通知用户（10分钟倒计时）
- 自动保存工作并关机

### 3.3 CPU/GPU频率动态调节
**功能描述：**
- 根据负载动态调节频率
- 保持高boost响应能力
- 长时间平均功耗降低

---

## RAG系统规范

### 技术选型
- **框架**：GraphRAG（微软开源）
- **存储**：YouTu Graph（或兼容Neo4j的图数据库）
- **向量数据库**：ChromaDB / Milvus
- **LLM接口**：OpenAI API / 本地LLM（Ollama）

### 应用场景
1. 建筑能耗知识库查询
2. 设备故障诊断知识库
3. 交通规则与优化策略知识库
4. 系统使用文档智能问答

---

## 部署规划

### 阶段一：建筑智能节能（优先级最高）
- Week 1-2: 基础架构（天气API、传感器接口）
- Week 3-4: 数字孪生模型
- Week 5-6: 控制策略（MPC/RL）
- Week 7-8: 储能调度与照明优化

### 阶段二：交通节能系统
- Week 1-2: 车辆检测跟踪
- Week 3-4: 跨摄像头匹配
- Week 5-6: 信号优化
- Week 7-8: 充电桩管理

### 阶段三：计算机节能系统
- Week 1: 进程监控
- Week 2: LLM决策
- Week 3: 频率调节

---

## 技术栈总览

| 模块 | 推荐技术 | 开源替代方案 |
|-----|---------|-------------|
| 天气数据 | OpenWeatherMap API / WeatherAPI.com | 中国天气网API |
| 建筑能耗模拟 | EnergyPlus + Python API | google/sbsim |
| 强化学习 | Stable-Baselines3 / Ray RLlib | mechyai/RL-EmsPy |
| 目标检测/跟踪 | Ultralytics YOLOv8 | 内置 BoT-SORT/ByteTrack |
| 车辆Re-ID | FastReID / OSNet | bharath5673/StrongSORT-YOLO |
| 时序预测 | Prophet / NeuralForecast | LSTM/Transformer自研 |
| 异常检测 | PyOD / scikit-learn | Isolation Forest |
| 系统监控 | psutil / WMI (Windows) | GPUtil for GPU |
| 数据库 | PostgreSQL + TimescaleDB | InfluxDB (时序数据) |
| RAG系统 | GraphRAG (微软) | LangChain + ChromaDB |
| LLM | OpenAI API / 本地LLM | Ollama + Llama3 |
| 图数据库 | Neo4j / YouTu Graph | 兼容GraphRAG |

---

## 可直接使用的开源项目

| 功能 | 推荐项目 | 说明 |
|-----|---------|------|
| HVAC RL控制 | VectorInstitute/HV-Ai-C | Vector Institute的强化学习HVAC控制 |
| 建筑能耗模拟 | google/sbsim | Google的强化学习建筑模拟器 |
| 多目标跟踪 | bharath5673/StrongSORT-YOLO | YOLO + StrongSORT |
| 建筑自动化 | hues-platform/nestli | 建筑自动化基准测试平台 |
| BACnet集成 | bbartling/easy-aso | 自动化监督优化框架 |
| 车辆计数 | ahmetozlu/vehicle_counting_tensorflow | 车辆检测计数 |
| GraphRAG | microsoft/graphrag | 微软开源GraphRAG |

---

## 文档版本
- 创建日期：2026年3月19日
- 版本：v1.0
- 状态：需求确认完成，待技术栈详细设计
