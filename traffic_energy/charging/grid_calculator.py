#!/usr/bin/env python3
"""
电网压力计算器模块

实现电网压力监测、电压/频率状态评估，以及电网事件检测。

Example:
    >>> from traffic_energy.charging.grid_calculator import GridPressureCalculator, GridState
    >>> calculator = GridPressureCalculator()
    >>> state = calculator.calculate(voltage=220.0, frequency=50.0, load_factor=0.6)
    >>> print(f"电网压力指数: {state.pressure_index:.2f}")
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import time
import json

from shared.logger import setup_logger

logger = setup_logger("grid_calculator")


class GridEventType(Enum):
    """电网事件类型"""
    HIGH_LOAD = "high_load"
    VOLTAGE_DROP = "voltage_drop"
    VOLTAGE_RISE = "voltage_rise"
    FREQUENCY_DEVIATION = "frequency_deviation"
    GRID_INSTABILITY = "grid_instability"


class GridSeverity(Enum):
    """电网事件严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class GridStatus(Enum):
    """电网状态"""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class GridState:
    """电网状态
    
    Attributes:
        timestamp: 时间戳
        voltage: 电压 (V)
        frequency: 频率 (Hz)
        load_factor: 负载率 (0-1)
        voltage_deviation: 电压偏差 (%)
        frequency_deviation: 频率偏差 (%)
        pressure_index: 电网压力指数 (0-1)
        status: 状态 ("normal" | "warning" | "critical")
    """
    timestamp: datetime
    voltage: float
    frequency: float
    load_factor: float
    voltage_deviation: float = 0.0
    frequency_deviation: float = 0.0
    pressure_index: float = 0.0
    status: str = "normal"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "voltage": self.voltage,
            "frequency": self.frequency,
            "load_factor": self.load_factor,
            "voltage_deviation": self.voltage_deviation,
            "frequency_deviation": self.frequency_deviation,
            "pressure_index": self.pressure_index,
            "status": self.status
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GridState":
        """从字典创建"""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            voltage=data["voltage"],
            frequency=data["frequency"],
            load_factor=data["load_factor"],
            voltage_deviation=data.get("voltage_deviation", 0.0),
            frequency_deviation=data.get("frequency_deviation", 0.0),
            pressure_index=data.get("pressure_index", 0.0),
            status=data.get("status", "normal")
        )


@dataclass
class GridEvent:
    """电网事件
    
    Attributes:
        event_id: 事件ID
        event_type: 事件类型
        severity: 严重程度
        start_time: 开始时间
        estimated_end_time: 预计结束时间
        affected_area: 影响区域
        recommended_actions: 建议操作列表
        description: 事件描述
    """
    event_id: str
    event_type: str
    severity: str
    start_time: datetime
    affected_area: str
    estimated_end_time: Optional[datetime] = None
    recommended_actions: List[str] = field(default_factory=list)
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "severity": self.severity,
            "start_time": self.start_time.isoformat(),
            "estimated_end_time": self.estimated_end_time.isoformat() if self.estimated_end_time else None,
            "affected_area": self.affected_area,
            "recommended_actions": self.recommended_actions,
            "description": self.description
        }


class GridPressureCalculator:
    """电网压力计算器
    
    基于IEEE Std 1547-2018标准计算电网压力指数。
    
    计算公式:
    - 电压偏差: |V - V_nominal| / V_nominal
    - 频率偏差: |f - f_nominal| / f_nominal
    - 压力指数: w1*|电压偏差| + w2*|频率偏差| + w3*负载率
    
    Attributes:
        voltage_nominal: 标称电压 (V)
        frequency_nominal: 标称频率 (Hz)
        voltage_tolerance: 电压容差 (±7%)
        frequency_tolerance: 频率容差 (±0.5Hz)
        weights: 权重配置
        
    Example:
        >>> calculator = GridPressureCalculator(
        ...     voltage_nominal=220.0,
        ...     frequency_nominal=50.0,
        ...     weights={"voltage": 0.4, "frequency": 0.3, "load": 0.3}
        ... )
        >>> state = calculator.calculate(220.0, 50.0, 0.6)
    """
    
    def __init__(
        self,
        voltage_nominal: float = 220.0,
        frequency_nominal: float = 50.0,
        voltage_tolerance: float = 0.07,
        frequency_tolerance: float = 0.5,
        weights: Optional[Dict[str, float]] = None
    ) -> None:
        """初始化计算器
        
        Args:
            voltage_nominal: 标称电压 (V)
            frequency_nominal: 标称频率 (Hz)
            voltage_tolerance: 电压容差
            frequency_tolerance: 频率容差 (Hz)
            weights: 权重配置 {"voltage": float, "frequency": float, "load": float}
        """
        self.voltage_nominal = voltage_nominal
        self.frequency_nominal = frequency_nominal
        self.voltage_tolerance = voltage_tolerance
        self.frequency_tolerance = frequency_tolerance
        
        # 默认权重: 电压40%, 频率30%, 负载30%
        self.weights = weights or {
            "voltage": 0.4,
            "frequency": 0.3,
            "load": 0.3
        }
        
        # 验证权重和为1
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.001:
            logger.warning(f"权重和不等于1.0: {total_weight}，将自动归一化")
            for key in self.weights:
                self.weights[key] /= total_weight
        
        # 阈值配置
        self.warning_threshold = 0.3
        self.critical_threshold = 0.6
        
        logger.info(f"初始化电网压力计算器: 标称电压={voltage_nominal}V, 标称频率={frequency_nominal}Hz")
    
    def calculate(
        self,
        voltage: float,
        frequency: float,
        load_factor: float
    ) -> GridState:
        """计算电网状态
        
        Args:
            voltage: 当前电压 (V)
            frequency: 当前频率 (Hz)
            load_factor: 负载率 (0-1)
            
        Returns:
            电网状态对象
        """
        # 计算电压偏差 (归一化到0-1范围)
        voltage_dev = abs(voltage - self.voltage_nominal) / self.voltage_nominal
        
        # 计算频率偏差 (归一化到0-1范围，假设最大偏差5Hz)
        freq_dev = abs(frequency - self.frequency_nominal) / self.frequency_nominal
        
        # 加权计算压力指数
        pressure = (
            self.weights["voltage"] * voltage_dev +
            self.weights["frequency"] * freq_dev +
            self.weights["load"] * load_factor
        )
        
        # 限制压力指数在0-1范围内
        pressure = max(0.0, min(1.0, pressure))
        
        # 状态评估
        if pressure < self.warning_threshold:
            status = "normal"
        elif pressure < self.critical_threshold:
            status = "warning"
        else:
            status = "critical"
        
        return GridState(
            timestamp=datetime.now(),
            voltage=voltage,
            frequency=frequency,
            load_factor=load_factor,
            voltage_deviation=voltage_dev * 100,  # 转换为百分比
            frequency_deviation=freq_dev * 100,   # 转换为百分比
            pressure_index=pressure,
            status=status
        )
    
    def calculate_batch(
        self,
        measurements: List[Dict[str, float]]
    ) -> List[GridState]:
        """批量计算电网状态
        
        Args:
            measurements: 测量值列表，每项包含voltage, frequency, load_factor
            
        Returns:
            电网状态列表
        """
        states = []
        for m in measurements:
            state = self.calculate(
                voltage=m["voltage"],
                frequency=m["frequency"],
                load_factor=m["load_factor"]
            )
            states.append(state)
        return states
    
    def detect_events(
        self,
        current_state: GridState,
        previous_state: Optional[GridState] = None
    ) -> List[GridEvent]:
        """检测电网事件
        
        Args:
            current_state: 当前电网状态
            previous_state: 上一个电网状态
            
        Returns:
            检测到的事件列表
        """
        events = []
        timestamp = current_state.timestamp
        
        # 高负载事件
        if current_state.load_factor > 0.8:
            severity = "high" if current_state.load_factor > 0.9 else "medium"
            events.append(GridEvent(
                event_id=f"high_load_{int(time.time())}",
                event_type="high_load",
                severity=severity,
                start_time=timestamp,
                estimated_end_time=timestamp + timedelta(hours=2),
                affected_area="grid_zone_1",
                recommended_actions=[
                    "降低非关键负载",
                    "启动需求响应程序",
                    "推迟非紧急充电"
                ],
                description=f"电网负载过高: {current_state.load_factor*100:.1f}%"
            ))
        
        # 电压跌落事件
        if current_state.voltage < self.voltage_nominal * (1 - self.voltage_tolerance):
            severity = "high" if current_state.voltage < self.voltage_nominal * 0.9 else "medium"
            events.append(GridEvent(
                event_id=f"voltage_drop_{int(time.time())}",
                event_type="voltage_drop",
                severity=severity,
                start_time=timestamp,
                estimated_end_time=timestamp + timedelta(hours=1),
                affected_area="grid_zone_1",
                recommended_actions=[
                    "降低充电功率",
                    "启动无功补偿",
                    "检查电网设备"
                ],
                description=f"电压跌落: {current_state.voltage:.1f}V"
            ))
        
        # 电压升高事件
        if current_state.voltage > self.voltage_nominal * (1 + self.voltage_tolerance):
            severity = "high" if current_state.voltage > self.voltage_nominal * 1.1 else "medium"
            events.append(GridEvent(
                event_id=f"voltage_rise_{int(time.time())}",
                event_type="voltage_rise",
                severity=severity,
                start_time=timestamp,
                estimated_end_time=timestamp + timedelta(hours=1),
                affected_area="grid_zone_1",
                recommended_actions=[
                    "增加负载吸收",
                    "检查电压调节设备",
                    "启动储能放电"
                ],
                description=f"电压升高: {current_state.voltage:.1f}V"
            ))
        
        # 频率偏差事件
        if abs(current_state.frequency - self.frequency_nominal) > self.frequency_tolerance:
            freq_dev = abs(current_state.frequency - self.frequency_nominal)
            severity = "high" if freq_dev > 1.0 else "medium"
            events.append(GridEvent(
                event_id=f"freq_dev_{int(time.time())}",
                event_type="frequency_deviation",
                severity=severity,
                start_time=timestamp,
                estimated_end_time=timestamp + timedelta(minutes=30),
                affected_area="grid_zone_1",
                recommended_actions=[
                    "启动一次调频",
                    "调整发电出力",
                    "限制大功率负载"
                ],
                description=f"频率偏差: {current_state.frequency:.2f}Hz"
            ))
        
        # 电网不稳定事件（状态突变）
        if previous_state and previous_state.status != current_state.status:
            if current_state.status == "critical":
                events.append(GridEvent(
                    event_id=f"instability_{int(time.time())}",
                    event_type="grid_instability",
                    severity="high",
                    start_time=timestamp,
                    estimated_end_time=timestamp + timedelta(minutes=15),
                    affected_area="grid_zone_1",
                    recommended_actions=[
                        "立即降低所有可控负载",
                        "启动紧急响应程序",
                        "通知电网调度中心"
                    ],
                    description=f"电网状态恶化: {previous_state.status} -> {current_state.status}"
                ))
        
        return events
    
    def get_pressure_trend(
        self,
        states: List[GridState],
        window_size: int = 5
    ) -> Dict[str, Any]:
        """获取压力趋势
        
        Args:
            states: 历史状态列表
            window_size: 滑动窗口大小
            
        Returns:
            趋势分析结果
        """
        if len(states) < window_size:
            return {"trend": "insufficient_data", "slope": 0.0}
        
        recent_states = states[-window_size:]
        pressures = [s.pressure_index for s in recent_states]
        
        # 简单线性回归计算趋势
        n = len(pressures)
        x = list(range(n))
        
        mean_x = sum(x) / n
        mean_y = sum(pressures) / n
        
        numerator = sum((x[i] - mean_x) * (pressures[i] - mean_y) for i in range(n))
        denominator = sum((x[i] - mean_x) ** 2 for i in range(n))
        
        slope = numerator / denominator if denominator != 0 else 0.0
        
        # 判断趋势
        if slope > 0.05:
            trend = "increasing"
        elif slope < -0.05:
            trend = "decreasing"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "slope": slope,
            "current_pressure": pressures[-1],
            "avg_pressure": mean_y,
            "max_pressure": max(pressures),
            "min_pressure": min(pressures)
        }
    
    def calibrate_thresholds(
        self,
        historical_states: List[GridState],
        target_normal_ratio: float = 0.7
    ) -> Dict[str, float]:
        """校准阈值
        
        根据历史数据自动调整warning和critical阈值
        
        Args:
            historical_states: 历史状态列表
            target_normal_ratio: 目标正常状态比例
            
        Returns:
            新的阈值配置
        """
        if not historical_states:
            return {"warning": self.warning_threshold, "critical": self.critical_threshold}
        
        pressures = [s.pressure_index for s in historical_states]
        pressures.sort()
        
        n = len(pressures)
        
        # 根据目标比例设置阈值
        warning_idx = int(n * target_normal_ratio)
        critical_idx = int(n * (target_normal_ratio + (1 - target_normal_ratio) / 2))
        
        new_warning = pressures[min(warning_idx, n - 1)]
        new_critical = pressures[min(critical_idx, n - 1)]
        
        # 确保阈值合理
        new_warning = max(0.1, min(0.5, new_warning))
        new_critical = max(new_warning + 0.1, min(0.9, new_critical))
        
        logger.info(f"阈值校准: warning={new_warning:.3f}, critical={new_critical:.3f}")
        
        return {
            "warning": new_warning,
            "critical": new_critical
        }


# 模拟数据生成器（用于开发和测试）
class GridDataSimulator:
    """电网数据模拟器
    
    用于开发和测试阶段的电网数据模拟。
    """
    
    def __init__(
        self,
        voltage_nominal: float = 220.0,
        frequency_nominal: float = 50.0,
        noise_level: float = 0.02
    ) -> None:
        """初始化模拟器
        
        Args:
            voltage_nominal: 标称电压
            frequency_nominal: 标称频率
            noise_level: 噪声水平
        """
        self.voltage_nominal = voltage_nominal
        self.frequency_nominal = frequency_nominal
        self.noise_level = noise_level
        self._step = 0
        
        import random
        self._random = random.Random(42)  # 固定种子以便复现
    
    def generate(self) -> Dict[str, float]:
        """生成模拟数据
        
        Returns:
            包含voltage, frequency, load_factor的字典
        """
        import math
        
        self._step += 1
        
        # 模拟日负荷曲线
        hour = (self._step % 1440) / 60  # 模拟24小时
        base_load = 0.4 + 0.3 * math.sin((hour - 6) * math.pi / 12)  # 日负荷曲线
        
        # 添加随机波动
        load_noise = self._random.uniform(-0.1, 0.1)
        load_factor = max(0.1, min(0.95, base_load + load_noise))
        
        # 电压随负载变化
        voltage_drop = (load_factor - 0.5) * 10  # 负载越高，电压越低
        voltage_noise = self._random.uniform(-2, 2)
        voltage = self.voltage_nominal + voltage_drop + voltage_noise
        
        # 频率随负载变化
        freq_drop = (load_factor - 0.5) * 0.2
        freq_noise = self._random.uniform(-0.1, 0.1)
        frequency = self.frequency_nominal + freq_drop + freq_noise
        
        return {
            "voltage": round(voltage, 2),
            "frequency": round(frequency, 2),
            "load_factor": round(load_factor, 3)
        }
    
    def generate_batch(self, count: int) -> List[Dict[str, float]]:
        """批量生成模拟数据
        
        Args:
            count: 生成数量
            
        Returns:
            模拟数据列表
        """
        return [self.generate() for _ in range(count)]


if __name__ == "__main__":
    # 简单测试
    calculator = GridPressureCalculator()
    
    # 正常状态
    state1 = calculator.calculate(220.0, 50.0, 0.5)
    print(f"正常状态: {state1}")
    
    # 警告状态
    state2 = calculator.calculate(205.0, 49.5, 0.75)
    print(f"警告状态: {state2}")
    
    # 紧急状态
    state3 = calculator.calculate(195.0, 48.5, 0.9)
    print(f"紧急状态: {state3}")
    
    # 检测事件
    events = calculator.detect_events(state3, state2)
    print(f"检测到 {len(events)} 个事件")
    for event in events:
        print(f"  - {event.event_type}: {event.description}")
