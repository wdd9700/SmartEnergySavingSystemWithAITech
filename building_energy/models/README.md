# 建筑设备异常检测模块

基于PyOD库的建筑设备异常检测系统，支持多种检测算法和HVAC系统集成。

## 功能特性

- **多种检测算法**: Isolation Forest、AutoEncoder、LOF、HBOS
- **HVAC专用检测器**: 针对暖通空调系统优化的检测器
- **实时告警机制**: 自动异常检测和告警生成
- **模型持久化**: 支持模型保存和加载
- **完整类型注解**: 代码可读性强，易于维护

## 安装依赖

```bash
pip install pyod scikit-learn scipy numpy
```

## 快速开始

### 1. 基础使用

```python
import numpy as np
from building_energy.models.anomaly_detector import AnomalyDetector

# 创建检测器
detector = AnomalyDetector(
    algorithm='iforest',
    contamination=0.1,
    alert_threshold=0.8
)

# 准备训练数据 (n_samples, n_features)
train_data = np.random.randn(200, 5)

# 训练模型
detector.fit(train_data)

# 预测异常
test_data = np.random.randn(50, 5)
predictions = detector.predict(test_data)  # 1表示异常，0表示正常

# 获取异常概率
proba = detector.predict_proba(test_data)
```

### 2. HVAC系统监控

```python
from building_energy.models.anomaly_detector import HVACAnomalyDetector

# 创建HVAC专用检测器
detector = HVACAnomalyDetector(
    algorithm='iforest',
    contamination=0.1,
    temp_threshold=2.0,    # 温度偏差阈值
    power_threshold=3.0    # 功耗异常阈值
)

# HVAC数据格式: [outdoor_temp, indoor_temp, indoor_humidity, hvac_power, 
#                setpoint_temp, occupancy, hour, solar_radiation]
hvac_data = np.random.randn(300, 8)

# 训练
detector.fit(hvac_data)

# 实时监控
result = detector.monitor(hvac_data[-10:])
print(result['system_status'])  # 系统状态
print(result['active_alerts'])  # 活动告警
```

### 3. 模型保存与加载

```python
# 保存模型
detector.save('models/anomaly_detector.pkl')

# 加载模型
detector = AnomalyDetector()
detector.load('models/anomaly_detector.pkl')

# 继续预测
predictions = detector.predict(new_data)
```

### 4. 告警处理

```python
# 获取所有告警
alerts = detector.get_alerts()

# 按严重程度过滤
critical_alerts = detector.get_alerts(severity='critical')

# 获取最近1小时的告警
from datetime import datetime, timedelta
recent_alerts = detector.get_alerts(
    since=datetime.now() - timedelta(hours=1)
)

# 清空告警历史
detector.clear_alerts()
```

## API参考

### AnomalyDetector

#### 构造函数

```python
AnomalyDetector(
    algorithm: str = 'iforest',      # 检测算法
    contamination: float = 0.1,      # 异常比例估计
    alert_threshold: float = 0.8,    # 告警阈值
    feature_names: List[str] = None, # 特征名称
    model_params: Dict = None        # 模型参数
)
```

#### 方法

| 方法 | 说明 |
|------|------|
| `fit(X)` | 训练模型 |
| `predict(X)` | 预测异常标签 (0/1) |
| `predict_proba(X)` | 预测异常概率 |
| `decision_function(X)` | 计算异常分数 |
| `save(path)` | 保存模型 |
| `load(path)` | 加载模型 |
| `get_alerts()` | 获取告警历史 |
| `clear_alerts()` | 清空告警 |
| `get_model_info()` | 获取模型信息 |

### HVACAnomalyDetector

HVAC专用检测器，继承自`AnomalyDetector`。

#### 额外方法

| 方法 | 说明 |
|------|------|
| `monitor(data)` | 监控HVAC系统状态 |

## 支持的算法

| 算法 | 说明 | 适用场景 |
|------|------|----------|
| `iforest` | 孤立森林 | 通用场景，计算效率高 |
| `autoencoder` | 自编码器 | 复杂模式检测 |
| `lof` | 局部异常因子 | 密度差异检测 |
| `hbos` | 直方图异常检测 | 大数据集，速度快 |

## 告警严重程度

- **critical**: 异常分数 >= 0.9
- **high**: 异常分数 >= 0.7
- **medium**: 异常分数 >= 0.5
- **low**: 异常分数 < 0.5

## 配置示例

```python
# 从配置创建检测器
from building_energy.models.anomaly_detector import create_detector_from_config

config = {
    'type': 'hvac',
    'algorithm': 'iforest',
    'contamination': 0.1,
    'alert_threshold': 0.8,
    'temp_threshold': 2.0,
    'power_threshold': 3.0
}

detector = create_detector_from_config(config)
```

## 测试

```bash
# 运行单元测试
python tests/test_anomaly_detector.py

# 使用pytest
pytest tests/test_anomaly_detector.py -v
```

## 注意事项

1. 训练数据至少需要10个样本
2. 异常比例`contamination`应在(0.0, 0.5]范围内
3. AutoEncoder算法需要PyTorch支持
4. 建议定期重新训练模型以适应设备老化

## 许可证

MIT License
