"""
预测结果监控器

负责收集PINN预测结果与实际传感器数据，创建预测-实际数据对。
支持PINN模型降级方案（当PINN不可用时使用简化热力学模型）。
"""

import logging
from typing import List, Optional, Dict, Any, Protocol
from dataclasses import dataclass
from datetime import datetime
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PredictionActualPair:
    """预测与实际数据对"""
    timestamp: datetime
    predicted_temp: float      # PINN预测温度 (°C)
    actual_temp: float         # 传感器实际温度 (°C)
    predicted_humidity: float  # PINN预测湿度 (%)
    actual_humidity: float     # 传感器实际湿度 (%)
    predicted_power: float     # PINN预测功耗 (kW)
    actual_power: float        # 电表实际功耗 (kW)
    zone_id: str              # 区域ID
    outdoor_temp: float       # 室外温度 (°C)
    occupancy: int            # 人员数量
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'predicted_temp': self.predicted_temp,
            'actual_temp': self.actual_temp,
            'predicted_humidity': self.predicted_humidity,
            'actual_humidity': self.actual_humidity,
            'predicted_power': self.predicted_power,
            'actual_power': self.actual_power,
            'zone_id': self.zone_id,
            'outdoor_temp': self.outdoor_temp,
            'occupancy': self.occupancy
        }


class SensorInterface(Protocol):
    """传感器接口协议"""
    
    def read_temperature(self, zone_id: str) -> float:
        """读取指定区域的温度"""
        ...
    
    def read_humidity(self, zone_id: str) -> float:
        """读取指定区域的湿度"""
        ...
    
    def read_power(self, device_id: str) -> float:
        """读取设备功耗"""
        ...
    
    def read_outdoor_temp(self) -> float:
        """读取室外温度"""
        ...
    
    def read_occupancy(self, zone_id: str) -> int:
        """读取区域人员数量"""
        ...


class PINNModel(Protocol):
    """PINN模型接口协议"""
    
    def predict(self, conditions: Dict[str, Any]) -> Dict[str, float]:
        """
        预测室内环境参数
        
        Args:
            conditions: 包含室外温度、设定温度、人员数量等条件的字典
            
        Returns:
            包含预测温度、湿度、功耗的字典
        """
        ...


class SimplifiedThermalModel:
    """
    简化热力学模型 (PINN降级方案)
    
    当PINN模型不可用时，使用简化的热力学方程进行预测。
    基于稳态热平衡方程：Q_hvac + Q_internal + Q_external = 0
    """
    
    def __init__(
        self,
        room_area: float = 50.0,          # 房间面积 (m²)
        room_height: float = 3.0,          # 房间高度 (m)
        wall_u_value: float = 0.5,         # 墙体U值 (W/m²K)
        window_u_value: float = 2.5,       # 窗户U值 (W/m²K)
        window_area_ratio: float = 0.2,    # 窗户面积比例
        cop_cooling: float = 3.0,          # 制冷COP
        cop_heating: float = 3.5,          # 制热COP
        internal_gain_per_person: float = 100.0  # 人均内部得热 (W)
    ):
        self.room_volume = room_area * room_height  # 房间体积 (m³)
        self.wall_area = room_area * 4 * room_height * (1 - window_area_ratio)
        self.window_area = room_area * 4 * room_height * window_area_ratio
        self.wall_u_value = wall_u_value
        self.window_u_value = window_u_value
        self.cop_cooling = cop_cooling
        self.cop_heating = cop_heating
        self.internal_gain_per_person = internal_gain_per_person
        
        # 空气热容 (J/K)
        self.air_thermal_capacity = self.room_volume * 1.225 * 1005
        
        logger.info("SimplifiedThermalModel initialized (PINN fallback)")
    
    def predict(self, conditions: Dict[str, Any]) -> Dict[str, float]:
        """
        预测室内环境参数
        
        Args:
            conditions: 包含以下键的字典:
                - outdoor_temp: 室外温度 (°C)
                - setpoint_temp: 设定温度 (°C)
                - occupancy: 人员数量
                - mode: 'cooling' 或 'heating'
                
        Returns:
            包含预测温度、湿度、功耗的字典
        """
        outdoor_temp = conditions.get('outdoor_temp', 30.0)
        setpoint_temp = conditions.get('setpoint_temp', 24.0)
        occupancy = conditions.get('occupancy', 0)
        mode = conditions.get('mode', 'cooling')
        
        # 计算热负荷
        # 1. 通过围护结构的传热
        q_wall = self.wall_u_value * self.wall_area * (outdoor_temp - setpoint_temp)
        q_window = self.window_u_value * self.window_area * (outdoor_temp - setpoint_temp)
        q_envelope = q_wall + q_window
        
        # 2. 内部得热 (人员)
        q_internal = occupancy * self.internal_gain_per_person
        
        # 3. 总热负荷
        q_total = q_envelope + q_internal
        
        # 计算所需HVAC功率
        if mode == 'cooling' and q_total > 0:
            hvac_power = q_total / self.cop_cooling / 1000  # 转换为kW
        elif mode == 'heating' and q_total < 0:
            hvac_power = abs(q_total) / self.cop_heating / 1000
        else:
            hvac_power = 0.5  # 最小运行功率
        
        # 预测室内湿度 (简化模型)
        # 假设空调除湿能力固定
        predicted_humidity = 50.0 + (outdoor_temp - setpoint_temp) * 0.5
        predicted_humidity = max(30.0, min(70.0, predicted_humidity))
        
        return {
            'temperature': setpoint_temp,
            'humidity': predicted_humidity,
            'power': hvac_power
        }


class PredictorMonitor:
    """
    预测结果监控器
    
    负责收集PINN预测结果与实际传感器数据，创建预测-实际数据对。
    支持历史数据缓存和滑动窗口管理。
    
    Attributes:
        pinn: PINN模型或降级方案
        sensors: 传感器接口
        history: 历史数据对列表
        max_history_size: 最大历史记录数
    
    Example:
        >>> monitor = PredictorMonitor(pinn_model, sensor_interface)
        >>> pair = monitor.collect('zone_1')
        >>> recent_history = monitor.get_recent_history(hours=24)
    """
    
    def __init__(
        self,
        pinn_model: Optional[PINNModel] = None,
        sensor_interface: Optional[SensorInterface] = None,
        max_history_size: int = 168,  # 默认保存7天数据(每小时)
        zone_configs: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """
        初始化预测监控器
        
        Args:
            pinn_model: PINN模型实例，如果为None则使用降级方案
            sensor_interface: 传感器接口实例
            max_history_size: 最大历史记录数
            zone_configs: 区域配置字典，包含各区域的房间参数
        """
        # 如果PINN模型不可用，使用降级方案
        if pinn_model is None:
            logger.warning("PINN model not provided, using simplified thermal model")
            self.pinn = SimplifiedThermalModel()
            self.using_fallback = True
        else:
            self.pinn = pinn_model
            self.using_fallback = False
        
        self.sensors = sensor_interface
        self.max_history_size = max_history_size
        self.zone_configs = zone_configs or {}
        
        # 使用deque实现自动淘汰的历史记录
        self.history: deque = deque(maxlen=max_history_size)
        
        # 区域到设备的映射
        self.zone_device_map: Dict[str, str] = {}
        
        logger.info(f"PredictorMonitor initialized (fallback={self.using_fallback})")
    
    def register_zone_device(self, zone_id: str, device_id: str) -> None:
        """
        注册区域对应的HVAC设备
        
        Args:
            zone_id: 区域ID
            device_id: HVAC设备ID
        """
        self.zone_device_map[zone_id] = device_id
        logger.debug(f"Registered device {device_id} for zone {zone_id}")
    
    def collect(self, zone_id: str) -> Optional[PredictionActualPair]:
        """
        收集预测与实际数据对
        
        Args:
            zone_id: 区域ID
            
        Returns:
            PredictionActualPair对象，如果数据收集失败返回None
        """
        try:
            timestamp = datetime.now()
            
            # 获取传感器实际数据
            if self.sensors:
                actual_temp = self.sensors.read_temperature(zone_id)
                actual_humidity = self.sensors.read_humidity(zone_id)
                outdoor_temp = self.sensors.read_outdoor_temp()
                occupancy = self.sensors.read_occupancy(zone_id)
                
                # 获取设备功耗
                device_id = self.zone_device_map.get(zone_id, zone_id)
                actual_power = self.sensors.read_power(device_id)
            else:
                # 模拟数据（用于测试）
                logger.warning("No sensor interface provided, using simulated data")
                actual_temp = 24.0 + np.random.normal(0, 0.5)
                actual_humidity = 50.0 + np.random.normal(0, 5)
                outdoor_temp = 30.0
                occupancy = 5
                actual_power = 2.5 + np.random.normal(0, 0.2)
            
            # 获取区域配置
            zone_config = self.zone_configs.get(zone_id, {})
            setpoint_temp = zone_config.get('setpoint_temp', 24.0)
            
            # 确定运行模式
            mode = 'cooling' if outdoor_temp > setpoint_temp else 'heating'
            
            # 获取PINN预测
            conditions = {
                'outdoor_temp': outdoor_temp,
                'setpoint_temp': setpoint_temp,
                'occupancy': occupancy,
                'mode': mode
            }
            
            prediction = self.pinn.predict(conditions)
            
            # 创建数据对
            pair = PredictionActualPair(
                timestamp=timestamp,
                predicted_temp=prediction.get('temperature', setpoint_temp),
                actual_temp=actual_temp,
                predicted_humidity=prediction.get('humidity', 50.0),
                actual_humidity=actual_humidity,
                predicted_power=prediction.get('power', 0.0),
                actual_power=actual_power,
                zone_id=zone_id,
                outdoor_temp=outdoor_temp,
                occupancy=occupancy
            )
            
            # 添加到历史记录
            self.history.append(pair)
            
            logger.debug(f"Collected data pair for zone {zone_id}: "
                        f"temp_deviation={pair.predicted_temp - pair.actual_temp:.2f}°C")
            
            return pair
            
        except Exception as e:
            logger.error(f"Failed to collect data for zone {zone_id}: {e}")
            return None
    
    def collect_all_zones(self) -> List[PredictionActualPair]:
        """
        收集所有注册区域的数据
        
        Returns:
            成功收集的数据对列表
        """
        results = []
        zones = list(self.zone_device_map.keys()) or ['default_zone']
        
        for zone_id in zones:
            pair = self.collect(zone_id)
            if pair:
                results.append(pair)
        
        logger.info(f"Collected data for {len(results)} zones")
        return results
    
    def get_history(self) -> List[PredictionActualPair]:
        """获取所有历史数据"""
        return list(self.history)
    
    def get_recent_history(
        self,
        hours: Optional[int] = None,
        zone_id: Optional[str] = None
    ) -> List[PredictionActualPair]:
        """
        获取近期历史数据
        
        Args:
            hours: 最近多少小时的数据，None表示全部
            zone_id: 指定区域，None表示所有区域
            
        Returns:
            符合条件的历史数据列表
        """
        history = list(self.history)
        
        if zone_id:
            history = [h for h in history if h.zone_id == zone_id]
        
        if hours:
            cutoff_time = datetime.now() - __import__('datetime').timedelta(hours=hours)
            history = [h for h in history if h.timestamp >= cutoff_time]
        
        return history
    
    def get_zone_history(self, zone_id: str) -> List[PredictionActualPair]:
        """获取指定区域的所有历史数据"""
        return [h for h in self.history if h.zone_id == zone_id]
    
    def clear_history(self) -> None:
        """清空历史记录"""
        self.history.clear()
        logger.info("History cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取监控统计信息
        
        Returns:
            包含统计信息的字典
        """
        if not self.history:
            return {
                'total_records': 0,
                'zones': [],
                'time_range': None
            }
        
        zones = set(h.zone_id for h in self.history)
        timestamps = [h.timestamp for h in self.history]
        
        return {
            'total_records': len(self.history),
            'zones': list(zones),
            'time_range': {
                'start': min(timestamps).isoformat(),
                'end': max(timestamps).isoformat()
            },
            'using_fallback': self.using_fallback
        }
