# 方向一：建筑智能节能系统 - 技术栈调研报告

## 调研日期
2026年3月19日

---

## 1. RAG系统技术栈

### 1.1 GraphRAG (微软开源)
**项目地址**: https://github.com/microsoft/graphrag  
**文档**: https://microsoft.github.io/graphrag/

**核心特性**:
- 使用知识图谱替代传统向量相似度搜索
- 支持全局搜索（Global Search）和局部搜索（Local Search）
- 使用Leiden算法进行层次聚类
- 自动生成社区摘要
- 支持DRIFT Search（动态推理）

**与YouTu Graph集成**:
- GraphRAG支持Neo4j兼容的图数据库
- YouTu Graph需要验证Cypher查询兼容性
- 备选：直接使用Neo4j Community Edition

**安装使用**:
```bash
pip install graphrag
graphrag init --root ./ragtest
graphrag index --root ./ragtest
graphrag query --root ./ragtest --method global "建筑能耗优化策略"
```

**应用场景**:
1. 建筑能耗知识库智能问答
2. 设备故障诊断知识检索
3. 控制策略文档理解

---

## 2. 建筑能耗模拟技术栈

### 2.1 EnergyPlus (DOE开发)
**项目地址**: https://github.com/NREL/EnergyPlus  
**官网**: https://energyplus.net/

**核心特性**:
- 美国能源部开发的建筑能耗模拟引擎
- 支持热负荷、冷负荷计算
- 支持HVAC系统建模
- 提供Python API (pyenergyplus)
- 支持实时数据交换（EMS）

**Python集成方式**:
```python
# 方式1: 使用EnergyPlus Python API
from pyenergyplus.api import EnergyPlusAPI
api = EnergyPlusAPI()
state = api.state_manager.new_state()
api.runtime.run_energyplus(state, ['-w', 'weather.epw', 'building.idf'])

# 方式2: 使用eppy库操作IDF文件
from eppy import modeleditor
from eppy.modeleditor import IDF
idf = IDF("building.idf")
```

### 2.2 eppy (EnergyPlus Python脚本)
**项目地址**: https://github.com/santoshphilip/eppy  
**文档**: https://eppy.readthedocs.io/

**核心特性**:
- 用Python脚本操作EnergyPlus IDF文件
- 支持批量修改建筑模型参数
- 支持读取EnergyPlus输出文件
- 支持并行运行多个模拟

**典型用法**:
```python
from eppy import modeleditor
from eppy.modeleditor import IDF

# 加载IDF文件
iddfile = "Energy+.idd"
fname1 = "building.idf"
IDF.setiddname(iddfile)
idf1 = IDF(fname1)

# 修改墙体材料
walls = idf1.idfobjects['BUILDINGSURFACE:DETAILED']
for wall in walls:
    if wall.Surface_Type == 'Wall':
        wall.Construction_Name = 'HighInsulationWall'

# 保存并运行
idf1.saveas('modified.idf')
idf1.run(output_directory='results')
```

### 2.3 Google sbsim (智能建筑仿真)
**项目地址**: https://github.com/google/sbsim

**核心特性**:
- Google开源的智能建筑强化学习仿真环境
- 基于真实建筑数据校准
- 提供6年真实建筑运行数据
- 与TensorFlow Agents集成
- 支持SAC、PPO等RL算法

**数据集**:
- TensorFlow Datasets: `smart_buildings`
- 包含3栋建筑6年历史数据
- 可直接用于训练RL模型

**使用示例**:
```python
from smart_control.environment import Environment
from smart_control.reinforcement_learning import SACAgent

# 创建环境
env = Environment.from_config('building_config.yaml')

# 训练SAC智能体
agent = SACAgent(env)
agent.train(num_iterations=100000)
```

---

## 3. 强化学习技术栈

### 3.1 Stable-Baselines3 (强烈推荐)
**项目地址**: https://github.com/DLR-RM/stable-baselines3  
**文档**: https://stable-baselines3.readthedocs.io/

**核心特性**:
- 基于PyTorch的强化学习库
- 支持PPO、SAC、TD3、A2C、DQN等算法
- 支持多环境并行训练 (VecEnv)
- 与Gymnasium环境完全兼容
- 提供Zoo预训练模型

**适用场景**:
- HVAC控制策略优化
- 储能充放电调度
- 照明控制优化
- 多区域温度控制

**安装**:
```bash
pip install stable-baselines3[extra]
```

**HVAC控制示例**:
```python
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

# 创建HVAC控制环境
class HVACEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # 动作空间: 温度设定点 [18, 26]°C
        self.action_space = gym.spaces.Box(low=18, high=26, shape=(1,))
        # 观察空间: [室外温度, 室内温度,  occupancy, 时间]
        self.observation_space = gym.spaces.Box(
            low=np.array([-20, 15, 0, 0]),
            high=np.array([45, 35, 100, 24]),
            dtype=np.float32
        )
    
    def step(self, action):
        # 执行控制动作
        setpoint = action[0]
        # ... HVAC仿真逻辑 ...
        
        # 奖励: 舒适度 + 能耗惩罚
        comfort_penalty = abs(self.indoor_temp - 22)  # 目标22°C
        energy_penalty = self.energy_consumption * 0.1
        reward = -comfort_penalty - energy_penalty
        
        return observation, reward, terminated, truncated, info

# 训练SAC智能体
env = DummyVecEnv([lambda: HVACEnv()])
model = SAC('MlpPolicy', env, verbose=1, 
            learning_rate=3e-4,
            buffer_size=100000,
            batch_size=256)
model.learn(total_timesteps=100000)
model.save('hvac_sac_model')
```

**优势**:
- ✅ **易用性**: 简洁API，快速上手
- ✅ **文档完善**: 详尽的文档和示例
- ✅ **社区活跃**: 持续更新和维护
- ✅ **生产就绪**: 经过大量实际项目验证

### 3.2 Ray RLlib (大规模分布式)
**项目地址**: https://github.com/ray-project/ray  
**文档**: https://docs.ray.io/en/latest/rllib/index.html

**核心特性**:
- 分布式强化学习框架
- 支持大规模并行训练 (数百个worker)
- 支持20+种算法（PPO、SAC、IMPALA、DreamerV3等）
- 与TensorFlow和PyTorch兼容
- 支持多智能体训练

**适用场景**:
- 大规模建筑集群控制
- 多智能体协同控制
- 需要分布式计算的场景
- 超参数大规模搜索

**与SB3对比**:

| 特性 | Stable-Baselines3 | Ray RLlib | 推荐场景 |
|------|-------------------|-----------|----------|
| 易用性 | ⭐⭐⭐ 优秀 | ⭐⭐ 中等 | SB3快速原型 |
| 分布式 | ❌ 不支持 | ⭐⭐⭐ 原生支持 | RLlib大规模训练 |
| 算法数量 | 10+ | 20+ | RLlib算法研究 |
| 资源占用 | 低 | 高 | SB3资源受限 |
| 调试难度 | 低 | 高 | SB3开发调试 |

**推荐**: 使用 **Stable-Baselines3** 进行快速原型开发和单节点训练，需要大规模分布式训练时迁移到 **Ray RLlib**。

---

## 4. 天气数据API

### 4.1 OpenWeatherMap API
**官网**: https://openweathermap.org/api

**免费额度**:
- 1000次/天（One Call API 3.0）
- 60次/分钟

**推荐API**:
- **One Call API 3.0**: 当前天气+48小时预报+历史数据
- **Air Pollution API**: 空气质量指数
- **Solar Irradiance API**: 太阳辐射数据（付费）

**Python调用**:
```python
import requests

API_KEY = "your_api_key"
lat, lon = 39.9042, 116.4074  # 北京坐标

url = f"https://api.openweathermap.org/data/3.0/onecall"
params = {
    'lat': lat,
    'lon': lon,
    'exclude': 'minutely',
    'appid': API_KEY,
    'units': 'metric'
}
response = requests.get(url, params=params)
data = response.json()
```

### 4.2 WeatherAPI.com
**官网**: https://www.weatherapi.com/

**免费额度**:
- 100万次/月
- 支持14天预报
- 支持历史数据查询

**优势**:
- 国内访问速度快
- 支持中文城市名
- 提供空气质量数据

---

## 5. 数字孪生建模技术

### 5.1 物理信息神经网络 (PINN)
**推荐库**:
- **NeuroDiffEq**: https://github.com/NeuroDiffGym/neurodiffeq
- **DeepXDE**: https://github.com/lululxvi/deepxde
- **SimNet** (NVIDIA): https://developer.nvidia.com/simnet

**应用场景**:
- 替代CFD进行温度场预测
- 建筑热传导建模
- 快速评估不同控制策略

### 5.2 代理模型 (Surrogate Model)
**推荐方法**:
- 高斯过程 (Gaussian Process)
- 神经网络代理模型
- 随机森林

**实现库**:
- **scikit-learn**: RandomForestRegressor
- **GPyTorch**: 高斯过程
- **PyTorch/TensorFlow**: 神经网络

---

## 6. 异常检测技术栈

### 6.1 PyOD (Python Outlier Detection)
**项目地址**: https://github.com/yzhao062/pyod

**支持算法**:
- Isolation Forest
- LOF (Local Outlier Factor)
- One-Class SVM
- AutoEncoder
- LSTM-based detection

**使用示例**:
```python
from pyod.models.iforest import IForest
from pyod.models.auto_encoder import AutoEncoder

# 训练Isolation Forest
clf = IForest(contamination=0.1)
clf.fit(X_train)

# 预测
y_pred = clf.predict(X_test)  # 0: 正常, 1: 异常
```

### 6.2 基于PID的残差分析
**实现思路**:
1. 建立空调系统的PID控制模型
2. 计算预测值与实际值的残差
3. 使用统计方法检测异常
4. 结合数字孪生模型进行故障定位

---

## 7. 推荐技术组合

### 7.1 方案A：学术研究导向（推荐）
| 模块 | 技术选型 | 理由 |
|-----|---------|------|
| 建筑模拟 | EnergyPlus + eppy | 行业标准，文档完善 |
| RL框架 | **Stable-Baselines3** ⭐ | 易用，适合研究，生产就绪 |
| 天气数据 | OpenWeatherMap | 免费额度充足 |
| 异常检测 | PyOD + 自定义PID模型 | 算法丰富 |
| RAG | GraphRAG + Neo4j | 微软背书 |
| 数字孪生 | PINN (DeepXDE) | 物理约束 |

### 7.2 方案B：工程部署导向
| 模块 | 技术选型 | 理由 |
|-----|---------|------|
| 建筑模拟 | Google sbsim | 已针对RL优化 |
| RL框架 | **Ray RLlib** ⭐ | 分布式支持，大规模训练 |
| 天气数据 | WeatherAPI.com | 国内访问快 |
| 异常检测 | 自研规则引擎 + ML | 可解释性强 |
| RAG | GraphRAG | 企业级 |
| 数字孪生 | 神经网络代理模型 | 推理速度快 |

### 7.3 技术选型决策树

```
项目规模和复杂度评估
├── 单建筑/小规模 (< 10个控制点)
│   ├── 快速原型 → Stable-Baselines3 + EnergyPlus
│   └── 生产部署 → SB3 + sbsim + 规则引擎
├── 多建筑/中等规模 (10-100个控制点)
│   ├── 集中式控制 → SB3 + 向量环境
│   └── 分布式控制 → Ray RLlib
└── 大规模集群 (> 100个控制点)
    └── Ray RLlib + 分布式训练
```

---

## 8. 参考开源项目

| 项目 | 地址 | 说明 |
|-----|------|------|
| HV-Ai-C | https://github.com/VectorInstitute/HV-Ai-C | Vector Institute的HVAC RL控制 |
| RL-EmsPy | https://github.com/mechyai/RL-EmsPy | EnergyPlus RL接口 |
| nestli | https://github.com/hues-platform/nestli | 建筑自动化基准测试 |
| easy-aso | https://github.com/bbartling/easy-aso | BACnet监督优化 |
| Demand-Response-RL | https://github.com/srmadani/Demand-Response-in-Commercial-Buildings | 需求响应RL |

---

## 9. 下一步行动建议

1. **立即开始**:
   - 安装EnergyPlus和eppy
   - 注册OpenWeatherMap API
   - 搭建GraphRAG基础环境

2. **第一周目标**:
   - 完成基础建筑模型（IDF文件）
   - 实现天气数据获取模块
   - 搭建RAG知识库框架

3. **技术验证**:
   - 验证EnergyPlus API与Python的集成
   - 测试GraphRAG与Neo4j的兼容性
   - 评估Stable-Baselines3在建筑控制场景的表现

---

## 文档版本
- 版本：v1.0
- 创建日期：2026年3月19日
- 状态：技术调研完成，待开发实施
