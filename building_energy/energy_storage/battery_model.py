"""
电池物理模型

实现电池储能系统的物理建模，包括SOC计算、充放电效率、容量衰减等。

参考:
- Battery model: https://doi.org/10.1016/j.jpowsour.2013.09.057
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class BatteryParams:
    """电池参数配置
    
    Attributes:
        capacity: 额定容量 (kWh), 范围: 5-100
        max_charge_power: 最大充电功率 (kW)
        max_discharge_power: 最大放电功率 (kW)
        efficiency: 充放电效率 (0-1)
        min_soc: 最小SOC (0-1), 默认0.1
        max_soc: 最大SOC (0-1), 默认0.9
        degradation_rate: 年衰减率 (0-1), 默认0.02 (2%/年)
        nominal_temperature: 额定工作温度 (°C), 默认25
        temperature_coefficient: 温度系数 (每°C影响效率), 默认0.005
    """
    capacity: float = 20.0
    max_charge_power: float = 10.0
    max_discharge_power: float = 10.0
    efficiency: float = 0.95
    min_soc: float = 0.1
    max_soc: float = 0.9
    degradation_rate: float = 0.02
    nominal_temperature: float = 25.0
    temperature_coefficient: float = 0.005
    
    def __post_init__(self):
        """验证参数有效性"""
        if not 5 <= self.capacity <= 100:
            raise ValueError(f"Capacity must be between 5-100 kWh, got {self.capacity}")
        if not 0 < self.efficiency <= 1:
            raise ValueError(f"Efficiency must be between 0-1, got {self.efficiency}")
        if not 0 <= self.min_soc < self.max_soc <= 1:
            raise ValueError(f"Invalid SOC range: {self.min_soc} - {self.max_soc}")
        if self.max_charge_power <= 0 or self.max_discharge_power <= 0:
            raise ValueError("Max charge/discharge power must be positive")


@dataclass
class BatteryState:
    """电池状态
    
    Attributes:
        soc: 当前SOC (0-1)
        current_power: 当前功率 (正为充电, 负为放电, kW)
        temperature: 电池温度 (°C)
        cycle_count: 循环次数
        health: 健康度 (0-1)
        last_update: 上次更新时间
    """
    soc: float = 0.5
    current_power: float = 0.0
    temperature: float = 25.0
    cycle_count: int = 0
    health: float = 1.0
    last_update: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """验证状态有效性"""
        if not 0 <= self.soc <= 1:
            raise ValueError(f"SOC must be between 0-1, got {self.soc}")
        if not 0 <= self.health <= 1:
            raise ValueError(f"Health must be between 0-1, got {self.health}")


class BatteryModel:
    """电池物理模型
    
    模拟电池的充放电行为、SOC变化、容量衰减等物理特性。
    
    Attributes:
        params: 电池参数
        state: 当前电池状态
    
    Example:
        >>> params = BatteryParams(capacity=20.0, max_charge_power=10.0)
        >>> battery = BatteryModel(params)
        >>> # 充电1小时，5kW功率
        >>> energy_charged = battery.charge(5.0, 1.0)
        >>> print(f"Charged: {energy_charged:.2f} kWh, SOC: {battery.state.soc:.2%}")
    """
    
    def __init__(self, params: BatteryParams, initial_soc: float = 0.5):
        """
        初始化电池模型
        
        Args:
            params: 电池参数配置
            initial_soc: 初始SOC (0-1)
        """
        self.params = params
        self.state = BatteryState(soc=initial_soc)
        self._cycle_accumulator = 0.0  # 用于计算等效循环次数
        
        logger.info(f"Battery initialized: {params.capacity}kWh, SOC={initial_soc:.1%}")
    
    def charge(self, power: float, duration: float) -> float:
        """充电操作
        
        Args:
            power: 充电功率 (kW), 必须为正
            duration: 充电时长 (小时)
        
        Returns:
            实际充入电量 (kWh)
        
        Raises:
            ValueError: 如果功率为负或超出限制
        """
        if power < 0:
            raise ValueError("Charge power must be positive")
        
        # 限制功率
        power = min(power, self.params.max_charge_power)
        
        # 计算考虑温度和效率的实际充电量
        temp_factor = self._get_temperature_factor()
        effective_efficiency = self.params.efficiency * temp_factor
        
        # 计算理论可充入电量
        available_capacity = (self.params.max_soc - self.state.soc) * self.params.capacity
        requested_energy = power * duration * effective_efficiency
        
        # 限制不超过可用容量
        actual_energy = min(requested_energy, available_capacity)
        actual_power = actual_energy / (duration * effective_efficiency) if duration > 0 else 0
        
        # 更新状态
        self.update(actual_power, duration)
        
        return actual_energy
    
    def discharge(self, power: float, duration: float) -> float:
        """放电操作
        
        Args:
            power: 放电功率 (kW), 必须为正
            duration: 放电时长 (小时)
        
        Returns:
            实际放出电量 (kWh)
        
        Raises:
            ValueError: 如果功率为负或超出限制
        """
        if power < 0:
            raise ValueError("Discharge power must be positive")
        
        # 限制功率
        power = min(power, self.params.max_discharge_power)
        
        # 计算考虑温度和效率的实际放电量
        temp_factor = self._get_temperature_factor()
        effective_efficiency = self.params.efficiency * temp_factor
        
        # 计算理论可放电量
        available_energy = (self.state.soc - self.params.min_soc) * self.params.capacity
        requested_energy = power * duration
        
        # 限制不超过可用电量
        actual_energy = min(requested_energy, available_energy)
        actual_power = actual_energy / duration if duration > 0 else 0
        
        # 更新状态 (放电功率为负)
        self.update(-actual_power, duration)
        
        # 返回实际放出的电量（考虑效率损失）
        return actual_energy * effective_efficiency
    
    def update(self, power: float, duration: float) -> None:
        """更新电池状态
        
        根据功率和时长更新SOC、循环次数等状态。
        
        Args:
            power: 功率 (kW), 正为充电, 负为放电
            duration: 时长 (小时)
        """
        if duration <= 0:
            return
        
        # 更新当前功率
        self.state.current_power = power
        
        # 计算能量变化
        energy_kwh = abs(power) * duration
        
        # 更新SOC
        if power > 0:  # 充电
            temp_factor = self._get_temperature_factor()
            effective_energy = energy_kwh * self.params.efficiency * temp_factor
            soc_change = effective_energy / self.params.capacity
            self.state.soc = min(self.state.soc + soc_change, self.params.max_soc)
        elif power < 0:  # 放电
            soc_change = energy_kwh / self.params.capacity
            self.state.soc = max(self.state.soc - soc_change, self.params.min_soc)
        
        # 更新循环计数器
        self._update_cycle_count(energy_kwh)
        
        # 更新健康度
        self._update_health()
        
        # 更新时间戳
        self.state.last_update = datetime.now()
    
    def _get_temperature_factor(self) -> float:
        """计算温度影响因子
        
        温度偏离额定温度时，效率会下降。
        
        Returns:
            温度因子 (0-1)
        """
        temp_diff = abs(self.state.temperature - self.params.nominal_temperature)
        factor = 1.0 - (temp_diff * self.params.temperature_coefficient)
        return max(0.8, min(1.0, factor))  # 最低80%效率
    
    def _update_cycle_count(self, energy_kwh: float) -> None:
        """更新等效循环次数
        
        Args:
            energy_kwh: 本次充放电量 (kWh)
        """
        # 一个完整循环 = 充满再放空
        cycle_fraction = energy_kwh / self.params.capacity
        self._cycle_accumulator += cycle_fraction
        
        # 每累积1个完整循环，计数器加1
        if self._cycle_accumulator >= 1.0:
            full_cycles = int(self._cycle_accumulator)
            self.state.cycle_count += full_cycles
            self._cycle_accumulator -= full_cycles
    
    def _update_health(self) -> None:
        """更新电池健康度 (SOH)
        
        基于循环次数和年衰减率计算健康度。
        """
        # 基于循环的衰减 (假设1000次循环后衰减到80%)
        cycle_degradation = self.state.cycle_count / 5000.0
        
        # 基于时间的衰减
        # 简化处理：假设已运行1年
        time_degradation = self.params.degradation_rate
        
        # 综合健康度
        self.state.health = max(0.6, 1.0 - cycle_degradation - time_degradation)
    
    def get_available_energy(self) -> float:
        """获取当前可用电量
        
        Returns:
            可用电量 (kWh)
        """
        return (self.state.soc - self.params.min_soc) * self.params.capacity * self.state.health
    
    def get_available_capacity(self) -> float:
        """获取当前可用容量
        
        Returns:
            可用容量 (kWh)
        """
        return (self.params.max_soc - self.state.soc) * self.params.capacity * self.state.health
    
    def get_max_charge_power(self) -> float:
        """获取当前最大充电功率
        
        考虑SOC限制和健康度。
        
        Returns:
            最大充电功率 (kW)
        """
        if self.state.soc >= self.params.max_soc:
            return 0.0
        return self.params.max_charge_power * self.state.health
    
    def get_max_discharge_power(self) -> float:
        """获取当前最大放电功率
        
        考虑SOC限制和健康度。
        
        Returns:
            最大放电功率 (kW)
        """
        if self.state.soc <= self.params.min_soc:
            return 0.0
        return self.params.max_discharge_power * self.state.health
    
    def set_temperature(self, temperature: float) -> None:
        """设置电池温度
        
        Args:
            temperature: 温度 (°C)
        """
        self.state.temperature = temperature
    
    def reset(self, soc: float = 0.5) -> None:
        """重置电池状态
        
        Args:
            soc: 重置后的SOC
        """
        self.state = BatteryState(soc=soc)
        self._cycle_accumulator = 0.0
        logger.info(f"Battery reset to SOC={soc:.1%}")
    
    def to_dict(self) -> dict:
        """转换为字典格式
        
        Returns:
            电池状态字典
        """
        return {
            "params": {
                "capacity": self.params.capacity,
                "max_charge_power": self.params.max_charge_power,
                "max_discharge_power": self.params.max_discharge_power,
                "efficiency": self.params.efficiency,
                "min_soc": self.params.min_soc,
                "max_soc": self.params.max_soc,
            },
            "state": {
                "soc": self.state.soc,
                "current_power": self.state.current_power,
                "temperature": self.state.temperature,
                "cycle_count": self.state.cycle_count,
                "health": self.state.health,
                "last_update": self.state.last_update.isoformat(),
            }
        }
