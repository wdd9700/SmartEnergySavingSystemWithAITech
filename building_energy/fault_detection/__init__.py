"""
预测偏差故障检测模块

该模块实现基于PINN预测结果与实际传感器数据对比的故障检测系统，
用于识别建筑HVAC系统中的"效应器"(空调设施)故障。

主要组件:
- PredictorMonitor: 预测结果监控器，收集PINN预测与实际传感器数据
- DeviationAnalyzer: 偏差分析器，计算偏差度量指标
- FaultLocator: 故障定位器，定位具体故障设备
- FaultAlerter: 故障告警器，分级告警系统

使用示例:
    >>> from building_energy.fault_detection import PredictorMonitor, DeviationAnalyzer, FaultLocator
    >>> monitor = PredictorMonitor(pinn_model, sensor_interface)
    >>> analyzer = DeviationAnalyzer()
    >>> locator = FaultLocator(device_registry)
"""

from .predictor_monitor import PredictorMonitor, PredictionActualPair
from .deviation_analyzer import DeviationAnalyzer, DeviationMetrics
from .fault_locator import FaultLocator, FaultDiagnosis
from .alerter import FaultAlerter, FaultAlert

__version__ = "1.0.0"
__all__ = [
    "PredictorMonitor",
    "PredictionActualPair",
    "DeviationAnalyzer",
    "DeviationMetrics",
    "FaultLocator",
    "FaultDiagnosis",
    "FaultAlerter",
    "FaultAlert",
]
