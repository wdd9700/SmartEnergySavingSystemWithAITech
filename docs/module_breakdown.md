# 模块拆解与依赖关系图

**项目**: AI节能系统 - 方向一(建筑能源) + 方向三(交通能源)  
**日期**: 2026年3月21日  
**状态**: 数字孪生可视化已在外部开发中

---

## 一、总体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AI节能系统 - 模块依赖图                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     方向一: 建筑能源智能管理                          │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                     │   │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │   │
│  │  │  Module 1A   │───→│  Module 1B   │───→│  Module 1C   │          │   │
│  │  │ PINN训练环境  │    │ 储能管理系统  │    │ 故障检测系统  │          │   │
│  │  │  (基础层)    │    │  (应用层)    │    │ (监控层)     │          │   │
│  │  └──────────────┘    └──────────────┘    └──────────────┘          │   │
│  │         │                   │                   │                  │   │
│  │         ↓                   ↓                   ↓                  │   │
│  │  ┌─────────────────────────────────────────────────────────────┐  │   │
│  │  │              Module 1D: 人因照明 + 预测性开关                   │  │   │
│  │  │                    (独立模块，可并行)                          │  │   │
│  │  └─────────────────────────────────────────────────────────────┘  │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     方向三: 交通能源智能管理                          │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                     │   │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │   │
│  │  │  Module 3A   │───→│  Module 3B   │───→│  Module 3C   │          │   │
│  │  │ 车牌颜色识别  │    │ 路径分析系统  │    │ LLM拥堵识别  │          │   │
│  │  │  (基础层)    │    │  (分析层)    │    │ (决策层)     │          │   │
│  │  └──────────────┘    └──────────────┘    └──────────────┘          │   │
│  │         │                   │                   │                  │   │
│  │         ↓                   ↓                   ↓                  │   │
│  │  ┌─────────────────────────────────────────────────────────────┐  │   │
│  │  │              Module 3D: 充电桩电网感知系统                      │  │   │
│  │  │              (独立模块，依赖电网数据接口)                       │  │   │
│  │  └─────────────────────────────────────────────────────────────┘  │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     共享基础设施 (已存在)                            │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │  • GraphRAG知识库 (knowledge/graph_rag.py)                          │   │
│  │  • YOLO检测 (traffic_energy/detection/)                             │   │
│  │  • 配置管理 (config/manager.py)                                      │   │
│  │  • 日志系统 (shared/logger.py)                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、模块详细说明

### 方向一: 建筑能源

#### Module 1A: PINN训练环境 (基础层)
**优先级**: P0 (最高)  
**依赖**: 无 (可立即开始)  
**被依赖**: Module 1B, Module 1C

**功能范围**:
- DeepXDE环境搭建
- 建筑热传导PDE定义
- 墙厚/体积/空调位置等参数建模
- 训练数据生成器 (CFD仿真数据)
- 模型训练与导出
- NPU适配接口

**技术栈**:
- DeepXDE (PINN框架)
- PyTorch (后端)
- NumPy (数值计算)

**交付物**:
- `building_energy/pinn/` 目录
- `thermal_pinn.py` - 热传导PINN模型
- `data_generator.py` - 训练数据生成
- `trainer.py` - 训练脚本
- `export.py` - 模型导出 (ONNX)

---

#### Module 1B: 储能管理系统 (应用层)
**优先级**: P1  
**依赖**: Module 1A (可选，可先用简化模型)  
**被依赖**: 无

**功能范围**:
- 电池储能模型
- 电价API集成 (分时电价)
- 电网频率/电压监测接口
- 充放电调度策略
- 多目标优化 (成本+舒适度+电网压力)

**技术栈**:
- OR-Tools (优化求解)
- 电价API (国家电网/第三方)
- 现有ConfigManager

**交付物**:
- `building_energy/energy_storage/` 目录
- `battery_model.py` - 电池模型
- `price_api.py` - 电价接口
- `scheduler.py` - 调度优化器
- `controller.py` - 储能控制器

---

#### Module 1C: 预测偏差故障检测 (监控层)
**优先级**: P1  
**依赖**: Module 1A (必须使用PINN预测结果)  
**被依赖**: 无

**功能范围**:
- 预测结果 vs 实际控制结果对比
- 偏差程度量化
- 历史拟合度评估
- "效应器"故障定位 (空调设施)
- 告警生成

**技术栈**:
- 现有AnomalyDetector (PyOD)
- 统计分析方法
- 时序异常检测

**交付物**:
- `building_energy/fault_detection/` 目录
- `predictor_monitor.py` - 预测监控
- `deviation_analyzer.py` - 偏差分析
- `fault_locator.py` - 故障定位
- `alerter.py` - 告警系统

---

#### Module 1D: 人因照明 + 预测性开关 (独立模块)
**优先级**: P2  
**依赖**: 无  
**被依赖**: 无

**功能范围**:
- 色温调节算法 (早晨冷光/傍晚暖光)
- 人员运动方向预测
- 提前开灯策略
- 与现有corridor_light集成

**技术栈**:
- 现有corridor_light系统
- 时序算法
- 简单运动预测

**交付物**:
- `building_energy/lighting/` 目录
- `circadian_rhythm.py` - 昼夜节律算法
- `motion_predictor.py` - 运动预测
- `adaptive_controller.py` - 自适应控制器

---

### 方向三: 交通能源

#### Module 3A: 车牌颜色识别 (基础层)
**优先级**: P0 (最高)  
**依赖**: 无 (可立即开始)  
**被依赖**: Module 3B

**功能范围**:
- YOLO车牌检测 (扩展现有检测器)
- 车牌区域裁剪
- HSV颜色空间分析
- 蓝牌/绿牌分类
- 结果集成到Detection数据类

**技术栈**:
- 现有YOLO检测器
- OpenCV (HSV转换)
- NumPy (颜色统计)

**交付物**:
- `traffic_energy/detection/plate_classifier.py`
- 修改 `vehicle_detector.py` 集成车牌检测
- `color_analyzer.py` - 颜色分析器

---

#### Module 3B: 路径分析系统 (分析层)
**优先级**: P1  
**依赖**: Module 3A (需要车辆类型数据)  
**被依赖**: Module 3C

**功能范围**:
- 轨迹聚类 (DBSCAN)
- 路径-时间图生成
- 车流量统计
- 拥堵程度计算
- 常见路径识别

**技术栈**:
- scikit-learn (DBSCAN)
- 现有VehicleTracker轨迹数据
- NumPy/Pandas (数据处理)

**交付物**:
- `traffic_energy/traffic_analysis/path_analyzer.py`
- `trajectory_clustering.py` - 轨迹聚类
- `flow_time_matrix.py` - 流量-时间矩阵
- `congestion_calculator.py` - 拥堵计算

---

#### Module 3C: LLM拥堵识别 (决策层)
**优先级**: P1  
**依赖**: Module 3B (需要拥堵数据)  
**被依赖**: 无

**功能范围**:
- VLM集成 (Qwen-VL或GPT-4V)
- 拥堵图像分析
- 拥堵原因识别 (事故/施工/信号灯等)
- RAG写入 (YouTu GraphRAG)
- 交警查询接口

**技术栈**:
- VLM API (Qwen-VL)
- YouTu GraphRAG (用户指定)
- 现有RAG基础设施

**交付物**:
- `traffic_energy/analysis/congestion_llm.py`
- `vlm_client.py` - VLM客户端
- `rag_writer.py` - RAG写入器
- `query_interface.py` - 查询接口

---

#### Module 3D: 充电桩电网感知 (独立模块)
**优先级**: P2  
**依赖**: 电网数据接口 (外部)  
**被依赖**: 无

**功能范围**:
- 电网压力计算公式实现
- 电压/频率监测接口
- 用户日程规划集成
- 动态充电调度

**技术栈**:
- 现有ChargingScheduler (OR-Tools)
- 电网API接口
- 用户行为模型

**交付物**:
- `traffic_energy/charging/grid_aware_scheduler.py`
- `grid_monitor.py` - 电网监测
- `user_schedule.py` - 用户日程
- `adaptive_charging.py` - 自适应充电

---

## 三、并行开发策略

### 第一波并行 (立即开始)
可完全独立开发，无依赖关系:

| 模块 | 开发者 | 预计工期 |
|-----|--------|---------|
| Module 1A: PINN训练环境 | Subagent 1 | 5-7天 |
| Module 1D: 人因照明 | Subagent 2 | 3-4天 |
| Module 3A: 车牌颜色识别 | Subagent 3 | 3-4天 |
| Module 3D: 充电桩电网感知 | Subagent 4 | 4-5天 |

### 第二波并行 (等待第一波完成)
依赖第一波模块:

| 模块 | 依赖 | 开发者 | 预计工期 |
|-----|------|--------|---------|
| Module 1B: 储能管理 | Module 1A (可选) | Subagent 5 | 4-5天 |
| Module 1C: 故障检测 | Module 1A (必须) | Subagent 6 | 4-5天 |
| Module 3B: 路径分析 | Module 3A (必须) | Subagent 7 | 4-5天 |

### 第三波并行 (等待第二波完成)
依赖第二波模块:

| 模块 | 依赖 | 开发者 | 预计工期 |
|-----|------|--------|---------|
| Module 3C: LLM拥堵识别 | Module 3B (必须) | Subagent 8 | 3-4天 |

---

## 四、接口契约

### Module 1A → Module 1C 接口
```python
# PINN预测结果输出
@dataclass
class ThermalPrediction:
    timestamp: datetime
    room_id: str
    temperature_map: np.ndarray  # 温度场分布
    confidence: float
    
class ThermalPINN:
    def predict(self, room_params: RoomParams, 
                external_conditions: WeatherData) -> ThermalPrediction:
        """预测房间温度分布"""
        pass
```

### Module 3A → Module 3B 接口
```python
# 车辆类型扩展示例
@dataclass
class Detection:
    # 现有字段...
    plate_color: str  # "blue" | "green" | "unknown"
    vehicle_power_type: str  # "fuel" | "electric" | "unknown"

# 轨迹数据
@dataclass
class VehicleTrajectory:
    track_id: int
    vehicle_type: str  # car/bus/truck
    power_type: str  # fuel/electric
    path_points: List[TrajectoryPoint]
    entry_time: datetime
    exit_time: datetime
```

---

## 五、风险与缓解

| 风险 | 影响 | 缓解措施 |
|-----|------|---------|
| DeepXDE NPU适配困难 | Module 1A延期 | 准备纯PyTorch备用方案 |
| VLM API不稳定 | Module 3C延期 | 准备规则引擎降级方案 |
| 电网数据接口不可用 | Module 3D延期 | 使用模拟数据继续开发 |
| 轨迹数据质量差 | Module 3B效果差 | 增加数据清洗模块 |

---

*文档结束*
