# Module 1C - 预测偏差故障检测系统

## 概述

预测偏差故障检测系统是建筑能源监控层的核心组件，负责对比PINN预测结果与实际传感器数据，识别HVAC系统中的"效应器"(空调设施)故障。

### 核心特性

- **零硬件成本**: 完全复用现有传感器，无需新增硬件
- **实时监测**: 检测延迟 < 5分钟
- **精准定位**: 定位精度到具体空调设备
- **智能分级**: 四级告警分级（低/中/高/严重）
- **高准确率**: 目标故障检测准确率 > 85%，误报率 < 5%

## 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                    预测偏差故障检测系统                        │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   PINN预测   │  │   传感器数据  │  │   历史数据   │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                 │              │
│         └────────┬────────┘                 │              │
│                  ▼                          │              │
│  ┌──────────────────────────────────┐      │              │
│  │     PredictorMonitor             │◄─────┘              │
│  │     (预测结果监控器)              │                     │
│  └──────────────┬───────────────────┘                     │
│                 ▼                                         │
│  ┌──────────────────────────────────┐                     │
│  │     DeviationAnalyzer            │                     │
│  │     (偏差分析器)                  │                     │
│  └──────────────┬───────────────────┘                     │
│                 ▼                                         │
│  ┌──────────────────────────────────┐                     │
│  │     FaultLocator                 │                     │
│  │     (故障定位器)                  │                     │
│  └──────────────┬───────────────────┘                     │
│                 ▼                                         │
│  ┌──────────────────────────────────┐                     │
│  │     FaultAlerter                 │                     │
│  │     (故障告警器)                  │                     │
│  └──────────────┬───────────────────┘                     │
│                 ▼                                         │
│  ┌──────────────────────────────────┐                     │
│  │     告警通知/日志/集成            │                     │
│  └──────────────────────────────────┘                     │
└─────────────────────────────────────────────────────────────┘
```

## 模块说明

### 1. PredictorMonitor (预测结果监控器)

负责收集PINN预测结果与实际传感器数据，创建预测-实际数据对。

**主要功能**:
- 对接PINN模型（支持降级方案）
- 对接传感器接口
- 数据对齐与预处理
- 历史数据缓存管理

**降级方案**: 当PINN模型不可用时，自动使用简化热力学模型

### 2. DeviationAnalyzer (偏差分析器)

计算预测值与实际值之间的偏差度量指标。

**核心指标**:
- MAE (平均绝对误差)
- RMSE (均方根误差)
- 最大偏差
- 相对偏差百分比

**功能特性**:
- 滑动窗口统计
- 历史拟合度评估
- 偏差趋势分析
- 显著偏差判定

### 3. FaultLocator (故障定位器)

根据偏差分析结果定位具体故障设备。

**故障判定规则**:
```
如果 室内温度偏差大 且 空调功耗异常:
    → 空调制冷/制热故障
    
如果 室内温度偏差大 且 空调功耗正常:
    → 传感器故障 或 房间密封性问题
    
如果 湿度偏差大:
    → 除湿/加湿设备故障
```

**支持的故障类型**:
- `ac_fault`: 空调制冷/制热故障
- `sensor_fault`: 传感器故障
- `sealing_fault`: 房间密封性问题
- `humidity_fault`: 除湿/加湿设备故障
- `power_fault`: 供电系统故障

### 4. FaultAlerter (故障告警器)

实现故障告警的分级、触发、抑制和恢复管理。

**告警分级**:
| 级别 | 条件 | 抑制期 | 通知间隔 |
|------|------|--------|----------|
| LOW | 轻微异常 | 60分钟 | 4小时 |
| MEDIUM | 一般故障 | 30分钟 | 2小时 |
| HIGH | 严重故障 | 15分钟 | 1小时 |
| CRITICAL | 紧急故障 | 5分钟 | 15分钟 |

**功能特性**:
- 告警抑制（避免重复告警）
- 告警确认机制
- 告警恢复跟踪
- 多渠道通知支持

## 快速开始

### 安装依赖

```bash
# 项目已包含的依赖
# - PyOD (异常检测)
# - NumPy/Pandas (数据处理)
# - scikit-learn (统计分析)
```

### 基本使用

```python
from building_energy.fault_detection import (
    PredictorMonitor, DeviationAnalyzer, FaultLocator, FaultAlerter
)
from building_energy.fault_detection.fault_locator import SimpleDeviceRegistry

# 创建设备注册表
registry = SimpleDeviceRegistry()
registry.register_device("hvac_001", "zone_1")

# 初始化组件
monitor = PredictorMonitor(pinn_model=None, sensor_interface=None)
monitor.register_zone_device("zone_1", "hvac_001")

analyzer = DeviationAnalyzer()
locator = FaultLocator(registry)
alerter = FaultAlerter()

# 添加通知处理器
from building_energy.fault_detection.alerter import console_notification_handler
alerter.add_notification_handler(console_notification_handler)

# 主循环
while True:
    # 收集数据
    pair = monitor.collect("zone_1")
    
    # 计算偏差
    history = monitor.get_history()
    metrics = analyzer.calculate_metrics(history)
    
    # 检查显著偏差
    if analyzer.is_deviation_significant(metrics):
        # 评估历史拟合度
        historical_fit = analyzer.assess_historical_fit(history)
        
        # 定位故障
        diagnosis = locator.locate_fault(pair, metrics, historical_fit)
        
        if diagnosis:
            # 触发告警
            alerter.alert(diagnosis)
    
    time.sleep(60)  # 每分钟检查一次
```

### 配置说明

配置文件位于 `config.yaml`，主要配置项:

```yaml
# 偏差阈值
analyzer:
  thresholds:
    temperature: 2.0       # 温度偏差阈值 (°C)
    humidity: 10.0         # 湿度偏差阈值 (%)
    power: 1.0             # 功耗偏差阈值 (kW)

# 告警抑制期
alerter:
  suppression_periods:
    low: 60        # 轻微告警抑制1小时
    medium: 30     # 一般告警抑制30分钟
    high: 15       # 严重告警抑制15分钟
    critical: 5    # 紧急告警抑制5分钟
```

## 故障诊断逻辑

### 故障判定条件

必须同时满足以下条件才会触发故障诊断:

1. **当前偏差显著**: |预测-实际| > 阈值
2. **历史拟合度良好**: 过去7天MAE < 0.5°C
3. **排除外部干扰**: 非极端天气等外部因素

### 置信度计算

故障置信度基于以下因素:
- 当前偏差与历史MAE的比值
- 历史拟合度 (0-1)
- 故障类型权重

```python
confidence = deviation_confidence * fit_weight * type_weight
```

### 故障类型权重

| 故障类型 | 权重 | 说明 |
|----------|------|------|
| 空调故障 | 0.9 | 最高优先级 |
| 湿度故障 | 0.8 | 高优先级 |
| 供电故障 | 0.75 | 中等优先级 |
| 传感器故障 | 0.7 | 需要进一步验证 |

## API参考

### PredictorMonitor

```python
class PredictorMonitor:
    def __init__(self, pinn_model=None, sensor_interface=None, max_history_size=168)
    def register_zone_device(self, zone_id: str, device_id: str) -> None
    def collect(self, zone_id: str) -> Optional[PredictionActualPair]
    def collect_all_zones(self) -> List[PredictionActualPair]
    def get_history(self) -> List[PredictionActualPair]
    def get_recent_history(self, hours=None, zone_id=None) -> List[PredictionActualPair]
```

### DeviationAnalyzer

```python
class DeviationAnalyzer:
    def __init__(self, window_size=168, temp_threshold=2.0, ...)
    def calculate_metrics(self, history, zone_id=None) -> DeviationMetrics
    def is_deviation_significant(self, metrics, check_historical_fit=True) -> bool
    def assess_historical_fit(self, history, zone_id=None) -> float
    def get_deviation_trend(self, history, window_size=24, zone_id=None) -> Dict
    def update_thresholds(self, **kwargs) -> None
```

### FaultLocator

```python
class FaultLocator:
    def __init__(self, device_registry=None, temp_fault_threshold=2.0, ...)
    def locate_fault(self, current, metrics, historical_fit, min_confidence=0.6) -> Optional[FaultDiagnosis]
    def register_device(self, device_id: str, zone_id: str, **kwargs) -> None
    def get_supported_fault_types(self) -> List[str]
```

### FaultAlerter

```python
class FaultAlerter:
    def __init__(self, anomaly_detector=None, suppression_periods=None, max_history_size=1000)
    def add_notification_handler(self, handler: Callable[[FaultAlert], None]) -> None
    def alert(self, diagnosis: FaultDiagnosis) -> Optional[FaultAlert]
    def acknowledge(self, alert_id: str, operator: str, notes=None) -> bool
    def resolve(self, alert_id: str, resolution_notes=None) -> bool
    def get_active_alerts(self, severity=None, device_id=None) -> List[FaultAlert]
    def get_alert_statistics(self) -> Dict[str, Any]
```

## 测试

运行测试:

```bash
# 运行偏差分析器测试
python -m pytest building_energy/fault_detection/tests/test_deviation.py -v

# 运行故障定位器测试
python -m pytest building_energy/fault_detection/tests/test_fault_locator.py -v

# 运行所有测试
python -m pytest building_energy/fault_detection/tests/ -v
```

## 性能指标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 故障检测准确率 | > 85% | 正确识别故障的比例 |
| 误报率 | < 5% | 错误报告故障的比例 |
| 检测延迟 | < 5分钟 | 从故障发生到检测的时间 |
| 定位精度 | 设备级 | 定位到具体空调设备 |

## 依赖关系

### 必须依赖

- **Module 1A (PINN模型)**: 提供预测结果
  - 如果不可用，将自动使用降级方案
- **现有传感器接口**: 提供实际传感器数据
- **现有AnomalyDetector**: 可选，用于增强异常检测

### 文件依赖

```
building_energy/fault_detection/
├── __init__.py              # 模块导出
├── predictor_monitor.py     # 依赖: pinn/ (可选)
├── deviation_analyzer.py    # 依赖: predictor_monitor
├── fault_locator.py         # 依赖: predictor_monitor, deviation_analyzer
├── alerter.py               # 依赖: fault_locator, models/anomaly_detector (可选)
└── config.yaml              # 配置文件
```

## 注意事项

### ⚠️ 重要提示

1. **PINN模型可用性**
   - 如果PINN模型不可用，系统将自动使用简化热力学模型
   - 降级方案的预测精度较低，可能影响故障检测准确性

2. **传感器数据质量**
   - 传感器可能存在噪声或漂移
   - 建议定期校准传感器

3. **阈值调优**
   - 阈值设置不当会导致误报或漏报
   - 建议使用历史数据进行阈值校准

4. **历史数据积累**
   - 系统需要至少1天的历史数据才能评估拟合度
   - 建议运行7天后启用故障检测功能

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| 1.0.0 | 2026-03-22 | 初始版本，实现核心功能 |

## 作者

智能故障诊断专家 Agent

## 许可证

本项目遵循主项目的许可证条款。
