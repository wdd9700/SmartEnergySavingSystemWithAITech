"""
故障定位器

负责根据偏差分析结果定位具体故障设备("效应器")，
实现基于规则的故障诊断逻辑。
"""

import logging
from typing import Optional, Dict, Any, List, Protocol
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

try:
    from .predictor_monitor import PredictionActualPair
    from .deviation_analyzer import DeviationMetrics
except ImportError:
    # 直接导入时回退到绝对导入
    from predictor_monitor import PredictionActualPair
    from deviation_analyzer import DeviationMetrics

logger = logging.getLogger(__name__)


class FaultType(Enum):
    """故障类型枚举"""
    AC_FAULT = "ac_fault"           # 空调制冷/制热故障
    SENSOR_FAULT = "sensor_fault"   # 传感器故障
    SEALING_FAULT = "sealing_fault" # 房间密封性问题
    HUMIDITY_FAULT = "humidity_fault"  # 除湿/加湿设备故障
    POWER_FAULT = "power_fault"     # 供电系统故障
    UNKNOWN = "unknown"             # 未知故障


class SeverityLevel(Enum):
    """严重级别枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FaultDiagnosis:
    """故障诊断结果"""
    fault_type: str           # 故障类型
    confidence: float         # 置信度 (0-1)
    affected_device: str      # 受影响的设备ID
    severity: str            # 严重级别
    description: str         # 故障描述
    recommended_action: str  # 建议措施
    timestamp: datetime
    details: Dict[str, Any]  # 详细诊断信息
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'fault_type': self.fault_type,
            'confidence': round(self.confidence, 3),
            'affected_device': self.affected_device,
            'severity': self.severity,
            'description': self.description,
            'recommended_action': self.recommended_action,
            'timestamp': self.timestamp.isoformat(),
            'details': self.details
        }


class DeviceRegistry(Protocol):
    """设备注册表协议"""
    
    def get_device_by_zone(self, zone_id: str) -> Optional[str]:
        """根据区域ID获取设备ID"""
        ...
    
    def get_device_info(self, device_id: str) -> Dict[str, Any]:
        """获取设备信息"""
        ...
    
    def get_all_devices(self) -> List[str]:
        """获取所有设备ID"""
        ...


class SimpleDeviceRegistry:
    """简单设备注册表实现"""
    
    def __init__(self):
        self._devices: Dict[str, Dict[str, Any]] = {}
        self._zone_map: Dict[str, str] = {}
    
    def register_device(
        self,
        device_id: str,
        zone_id: str,
        device_type: str = "hvac",
        capacity: float = 5.0,
        **kwargs
    ) -> None:
        """注册设备"""
        self._devices[device_id] = {
            'device_id': device_id,
            'zone_id': zone_id,
            'device_type': device_type,
            'capacity': capacity,
            **kwargs
        }
        self._zone_map[zone_id] = device_id
    
    def get_device_by_zone(self, zone_id: str) -> Optional[str]:
        """根据区域ID获取设备ID"""
        return self._zone_map.get(zone_id)
    
    def get_device_info(self, device_id: str) -> Dict[str, Any]:
        """获取设备信息"""
        return self._devices.get(device_id, {})
    
    def get_all_devices(self) -> List[str]:
        """获取所有设备ID"""
        return list(self._devices.keys())


class FaultLocator:
    """
    故障定位器
    
    根据偏差分析结果定位具体故障设备，实现基于规则的故障诊断逻辑。
    
    故障判定规则:
    1. 室内温度偏差大 + 空调功耗异常 → 空调制冷/制热故障
    2. 室内温度偏差大 + 空调功耗正常 → 传感器故障或房间密封性问题
    3. 湿度偏差大 → 除湿/加湿设备故障
    
    Attributes:
        devices: 设备注册表
        temp_fault_threshold: 温度故障阈值 (°C)
        humidity_fault_threshold: 湿度故障阈值 (%)
        power_deviation_threshold: 功耗偏差阈值 (kW)
    
    Example:
        >>> locator = FaultLocator(device_registry)
        >>> diagnosis = locator.locate_fault(pair, metrics, historical_fit)
        >>> if diagnosis:
        ...     print(f"Detected {diagnosis.fault_type} in {diagnosis.affected_device}")
    """
    
    def __init__(
        self,
        device_registry: Optional[DeviceRegistry] = None,
        temp_fault_threshold: float = 2.0,      # 温度故障阈值
        humidity_fault_threshold: float = 15.0,  # 湿度故障阈值
        power_deviation_threshold: float = 1.0   # 功耗偏差阈值
    ):
        """
        初始化故障定位器
        
        Args:
            device_registry: 设备注册表
            temp_fault_threshold: 温度故障阈值
            humidity_fault_threshold: 湿度故障阈值
            power_deviation_threshold: 功耗偏差阈值
        """
        self.devices = device_registry or SimpleDeviceRegistry()
        self.temp_fault_threshold = temp_fault_threshold
        self.humidity_fault_threshold = humidity_fault_threshold
        self.power_deviation_threshold = power_deviation_threshold
        
        # 故障计数器（用于置信度计算）
        self._fault_counters: Dict[str, int] = {}
        
        logger.info("FaultLocator initialized")
    
    def locate_fault(
        self,
        current: PredictionActualPair,
        metrics: DeviationMetrics,
        historical_fit: float,
        min_confidence: float = 0.6
    ) -> Optional[FaultDiagnosis]:
        """
        定位故障
        
        Args:
            current: 当前预测-实际数据对
            metrics: 偏差度量指标
            historical_fit: 历史拟合度 (0-1)
            min_confidence: 最小置信度阈值
            
        Returns:
            故障诊断结果，如无故障返回None
        """
        # 检查历史拟合度
        if historical_fit < 0.5:
            logger.warning(f"Historical fit too low ({historical_fit:.2f}), skipping fault detection")
            return None
        
        # 获取设备ID
        device_id = self.devices.get_device_by_zone(current.zone_id) or current.zone_id
        
        # 分析温度偏差
        temp_deviation = abs(current.predicted_temp - current.actual_temp)
        temp_analysis = self._analyze_temperature_deviation(current, metrics)
        
        # 分析湿度偏差
        humidity_deviation = abs(current.predicted_humidity - current.actual_humidity)
        humidity_analysis = self._analyze_humidity_deviation(current, metrics)
        
        # 分析功耗偏差
        power_analysis = self._analyze_power_deviation(current, metrics)
        
        # 综合诊断
        diagnosis = None
        confidence = 0.0
        
        # 优先级1: 温度偏差（最严重）
        if temp_deviation > self.temp_fault_threshold:
            if power_analysis['is_abnormal']:
                # 温度偏差大 + 功耗异常 = 空调故障
                confidence = self._calculate_confidence(
                    temp_deviation, metrics.temp_mae, historical_fit, 'ac'
                )
                if confidence >= min_confidence:
                    diagnosis = self._create_ac_fault_diagnosis(
                        device_id, current, metrics, confidence, temp_analysis
                    )
            else:
                # 温度偏差大 + 功耗正常 = 传感器或密封性问题
                confidence = self._calculate_confidence(
                    temp_deviation, metrics.temp_mae, historical_fit, 'sensor'
                )
                if confidence >= min_confidence:
                    diagnosis = self._create_sensor_fault_diagnosis(
                        device_id, current, metrics, confidence, temp_analysis
                    )
        
        # 优先级2: 湿度偏差
        elif humidity_deviation > self.humidity_fault_threshold:
            confidence = self._calculate_confidence(
                humidity_deviation, metrics.humidity_mae, historical_fit, 'humidity'
            )
            if confidence >= min_confidence:
                diagnosis = self._create_humidity_fault_diagnosis(
                    device_id, current, metrics, confidence, humidity_analysis
                )
        
        # 优先级3: 仅功耗异常
        elif power_analysis['is_abnormal'] and not temp_analysis['is_deviation']:
            confidence = self._calculate_confidence(
                power_analysis['deviation'], metrics.power_mae, historical_fit, 'power'
            )
            if confidence >= min_confidence:
                diagnosis = self._create_power_fault_diagnosis(
                    device_id, current, metrics, confidence, power_analysis
                )
        
        if diagnosis:
            logger.warning(f"Fault detected: {diagnosis.fault_type} in {device_id} "
                          f"(confidence={confidence:.2f})")
        
        return diagnosis
    
    def _analyze_temperature_deviation(
        self,
        pair: PredictionActualPair,
        metrics: DeviationMetrics
    ) -> Dict[str, Any]:
        """
        分析温度偏差原因
        
        Args:
            pair: 预测-实际数据对
            metrics: 偏差度量指标
            
        Returns:
            分析结果字典
        """
        deviation = pair.predicted_temp - pair.actual_temp
        abs_deviation = abs(deviation)
        
        # 判断偏差方向
        if deviation > 0:
            direction = "undercooling" if pair.outdoor_temp > pair.predicted_temp else "underheating"
        else:
            direction = "overcooling" if pair.outdoor_temp > pair.predicted_temp else "overheating"
        
        # 判断是否显著偏差
        is_deviation = abs_deviation > self.temp_fault_threshold
        
        # 评估偏差严重性
        if abs_deviation > 5.0:
            severity = SeverityLevel.CRITICAL
        elif abs_deviation > 3.0:
            severity = SeverityLevel.HIGH
        elif abs_deviation > 1.5:
            severity = SeverityLevel.MEDIUM
        else:
            severity = SeverityLevel.LOW
        
        return {
            'deviation': deviation,
            'abs_deviation': abs_deviation,
            'direction': direction,
            'is_deviation': is_deviation,
            'severity': severity,
            'outdoor_temp': pair.outdoor_temp,
            'occupancy': pair.occupancy
        }
    
    def _analyze_humidity_deviation(
        self,
        pair: PredictionActualPair,
        metrics: DeviationMetrics
    ) -> Dict[str, Any]:
        """分析湿度偏差原因"""
        deviation = pair.predicted_humidity - pair.actual_humidity
        abs_deviation = abs(deviation)
        
        is_deviation = abs_deviation > self.humidity_fault_threshold
        
        if deviation > 0:
            issue = "dehumidification_failure"  # 除湿不足
        else:
            issue = "humidification_failure"    # 加湿过度或除湿过度
        
        return {
            'deviation': deviation,
            'abs_deviation': abs_deviation,
            'is_deviation': is_deviation,
            'issue': issue
        }
    
    def _analyze_power_deviation(
        self,
        pair: PredictionActualPair,
        metrics: DeviationMetrics
    ) -> Dict[str, Any]:
        """分析功耗偏差原因"""
        deviation = pair.predicted_power - pair.actual_power
        abs_deviation = abs(deviation)
        
        is_abnormal = abs_deviation > self.power_deviation_threshold
        
        if deviation > 0:
            status = "under_consumption"  # 功耗偏低
        else:
            status = "over_consumption"   # 功耗偏高
        
        # 计算相对偏差
        if pair.predicted_power > 0.1:
            relative_dev = abs_deviation / pair.predicted_power * 100
        else:
            relative_dev = 0.0
        
        return {
            'deviation': deviation,
            'abs_deviation': abs_deviation,
            'relative_deviation': relative_dev,
            'is_abnormal': is_abnormal,
            'status': status
        }
    
    def _calculate_confidence(
        self,
        current_deviation: float,
        historical_mae: float,
        historical_fit: float,
        fault_type: str
    ) -> float:
        """
        计算故障置信度
        
        置信度基于:
        1. 当前偏差与历史MAE的比值
        2. 历史拟合度
        3. 故障类型权重
        """
        # 基于偏差的置信度
        if historical_mae > 0:
            deviation_confidence = min(current_deviation / (historical_mae + 0.5), 1.0)
        else:
            deviation_confidence = min(current_deviation / 3.0, 1.0)
        
        # 历史拟合度权重
        fit_weight = historical_fit
        
        # 故障类型权重
        type_weights = {
            'ac': 0.9,
            'sensor': 0.7,
            'humidity': 0.8,
            'power': 0.75
        }
        type_weight = type_weights.get(fault_type, 0.7)
        
        # 综合置信度
        confidence = deviation_confidence * fit_weight * type_weight
        
        # 确保在合理范围内
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence
    
    def _create_ac_fault_diagnosis(
        self,
        device_id: str,
        pair: PredictionActualPair,
        metrics: DeviationMetrics,
        confidence: float,
        analysis: Dict[str, Any]
    ) -> FaultDiagnosis:
        """创建空调故障诊断"""
        deviation = analysis['abs_deviation']
        direction = analysis['direction']
        
        # 确定严重级别
        if deviation > 5.0:
            severity = SeverityLevel.CRITICAL.value
        elif deviation > 3.0:
            severity = SeverityLevel.HIGH.value
        else:
            severity = SeverityLevel.MEDIUM.value
        
        # 生成描述和建议
        if direction == "undercooling":
            description = f"空调制冷能力不足，室内温度比预测高{deviation:.1f}°C"
            action = "检查制冷剂压力、压缩机运行状态、冷凝器清洁度"
        elif direction == "underheating":
            description = f"空调制热能力不足，室内温度比预测低{deviation:.1f}°C"
            action = "检查制热模式、四通阀、室外机除霜功能"
        else:
            description = f"空调温度控制异常，偏差{deviation:.1f}°C"
            action = "检查温控器设置、传感器校准、风机运行"
        
        return FaultDiagnosis(
            fault_type=FaultType.AC_FAULT.value,
            confidence=confidence,
            affected_device=device_id,
            severity=severity,
            description=description,
            recommended_action=action,
            timestamp=datetime.now(),
            details={
                'temperature_deviation': deviation,
                'direction': direction,
                'power_deviation': metrics.power_mae,
                'outdoor_temp': pair.outdoor_temp
            }
        )
    
    def _create_sensor_fault_diagnosis(
        self,
        device_id: str,
        pair: PredictionActualPair,
        metrics: DeviationMetrics,
        confidence: float,
        analysis: Dict[str, Any]
    ) -> FaultDiagnosis:
        """创建传感器故障诊断"""
        deviation = analysis['abs_deviation']
        
        # 判断是传感器问题还是密封性问题
        if pair.occupancy > 0 and deviation > 3.0:
            # 有人且偏差大，可能是密封性问题
            fault_type = FaultType.SEALING_FAULT
            description = f"房间密封性可能存在问题，温度偏差{deviation:.1f}°C但空调运行正常"
            action = "检查门窗密封、墙体保温、新风系统"
        else:
            fault_type = FaultType.SENSOR_FAULT
            description = f"温度传感器可能存在故障，读数偏差{deviation:.1f}°C"
            action = "校准或更换温度传感器"
        
        severity = SeverityLevel.MEDIUM.value if deviation < 4.0 else SeverityLevel.HIGH.value
        
        return FaultDiagnosis(
            fault_type=fault_type.value,
            confidence=confidence,
            affected_device=device_id,
            severity=severity,
            description=description,
            recommended_action=action,
            timestamp=datetime.now(),
            details={
                'temperature_deviation': deviation,
                'power_normal': True,
                'occupancy': pair.occupancy
            }
        )
    
    def _create_humidity_fault_diagnosis(
        self,
        device_id: str,
        pair: PredictionActualPair,
        metrics: DeviationMetrics,
        confidence: float,
        analysis: Dict[str, Any]
    ) -> FaultDiagnosis:
        """创建湿度故障诊断"""
        deviation = analysis['abs_deviation']
        issue = analysis['issue']
        
        if issue == "dehumidification_failure":
            description = f"除湿功能异常，湿度比预测高{deviation:.1f}%"
            action = "检查除湿器、冷凝水排放、制冷系统"
        else:
            description = f"湿度控制异常，偏差{deviation:.1f}%"
            action = "检查加湿器/除湿器运行状态"
        
        severity = SeverityLevel.MEDIUM.value if deviation < 20.0 else SeverityLevel.HIGH.value
        
        return FaultDiagnosis(
            fault_type=FaultType.HUMIDITY_FAULT.value,
            confidence=confidence,
            affected_device=device_id,
            severity=severity,
            description=description,
            recommended_action=action,
            timestamp=datetime.now(),
            details={
                'humidity_deviation': deviation,
                'issue_type': issue
            }
        )
    
    def _create_power_fault_diagnosis(
        self,
        device_id: str,
        pair: PredictionActualPair,
        metrics: DeviationMetrics,
        confidence: float,
        analysis: Dict[str, Any]
    ) -> FaultDiagnosis:
        """创建供电故障诊断"""
        status = analysis['status']
        relative_dev = analysis['relative_deviation']
        
        if status == "under_consumption":
            description = f"设备功耗异常偏低，比预测低{relative_dev:.1f}%"
            action = "检查设备运行状态、电源供应、电表读数"
        else:
            description = f"设备功耗异常偏高，比预测高{relative_dev:.1f}%"
            action = "检查设备负载、电机效率、过滤器堵塞"
        
        return FaultDiagnosis(
            fault_type=FaultType.POWER_FAULT.value,
            confidence=confidence,
            affected_device=device_id,
            severity=SeverityLevel.LOW.value,
            description=description,
            recommended_action=action,
            timestamp=datetime.now(),
            details={
                'power_deviation': analysis['abs_deviation'],
                'relative_deviation': relative_dev,
                'status': status
            }
        )
    
    def register_device(self, device_id: str, zone_id: str, **kwargs) -> None:
        """注册设备到注册表"""
        if isinstance(self.devices, SimpleDeviceRegistry):
            self.devices.register_device(device_id, zone_id, **kwargs)
            logger.info(f"Registered device {device_id} for zone {zone_id}")
    
    def get_supported_fault_types(self) -> List[str]:
        """获取支持的故障类型列表"""
        return [ft.value for ft in FaultType]
