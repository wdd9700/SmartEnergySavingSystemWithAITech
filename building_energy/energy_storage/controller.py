"""
储能控制器

主控制逻辑，协调电池模型、电价API和调度优化器。
提供事件响应机制和降级策略。
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Callable
from enum import Enum
from pathlib import Path

import yaml

try:
    from .battery_model import BatteryModel, BatteryParams
    from .price_api import PriceAPI, PriceSchedule
    from .scheduler import EnergyScheduler, Schedule, HVACForecast, OptimizationObjective
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from battery_model import BatteryModel, BatteryParams
    from price_api import PriceAPI, PriceSchedule
    from scheduler import EnergyScheduler, Schedule, HVACForecast, OptimizationObjective

logger = logging.getLogger(__name__)


class ControllerState(Enum):
    """控制器状态"""
    IDLE = "idle"
    CHARGING = "charging"
    DISCHARGING = "discharging"
    OPTIMIZING = "optimizing"
    EMERGENCY = "emergency"
    ERROR = "error"


@dataclass
class PriceEvent:
    """电价事件
    
    Attributes:
        timestamp: 事件发生时间
        event_type: 事件类型 ("price_spike", "price_drop", "peak_hour_start", etc.)
        price: 当前电价
        period: 时段类型
        message: 事件描述
    """
    timestamp: datetime
    event_type: str
    price: float
    period: str
    message: str = ""


@dataclass
class GridEvent:
    """电网事件
    
    Attributes:
        timestamp: 事件发生时间
        event_type: 事件类型 ("demand_response", "peak_shaving", "grid_stress", etc.)
        priority: 优先级 (1-5, 5最高)
        command: 指令 ("charge", "discharge", "idle")
        power_limit: 功率限制 (kW)
        duration: 持续时间 (分钟)
        message: 事件描述
    """
    timestamp: datetime
    event_type: str
    priority: int = 3
    command: str = "idle"
    power_limit: Optional[float] = None
    duration: int = 60
    message: str = ""


@dataclass
class ControllerMetrics:
    """控制器运行指标
    
    Attributes:
        total_cycles: 总调度周期数
        total_energy_charged: 总充电量 (kWh)
        total_energy_discharged: 总放电量 (kWh)
        total_cost_savings: 总节省费用 (元)
        grid_events_handled: 处理的电网事件数
        price_events_handled: 处理的电价事件数
        last_update: 上次更新时间
    """
    total_cycles: int = 0
    total_energy_charged: float = 0.0
    total_energy_discharged: float = 0.0
    total_cost_savings: float = 0.0
    grid_events_handled: int = 0
    price_events_handled: int = 0
    last_update: datetime = field(default_factory=datetime.now)


class StorageController:
    """储能控制器
    
    主控制逻辑，协调电池、电价和调度器。
    支持事件响应、异常处理和降级策略。
    
    Attributes:
        battery: 电池模型
        price_api: 电价API
        scheduler: 调度优化器
        state: 当前控制器状态
        metrics: 运行指标
    
    Example:
        >>> controller = StorageController.from_config("config.yaml")
        >>> controller.start()
        >>> # 运行一段时间后
        >>> controller.stop()
        >>> print(f"Total savings: {controller.metrics.total_cost_savings:.2f} 元")
    """
    
    def __init__(
        self,
        battery: BatteryModel,
        price_api: PriceAPI,
        scheduler: EnergyScheduler,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化控制器
        
        Args:
            battery: 电池模型
            price_api: 电价API
            scheduler: 调度优化器
            config: 配置参数
        """
        self.battery = battery
        self.price_api = price_api
        self.scheduler = scheduler
        self.config = config or {}
        
        # 状态
        self.state = ControllerState.IDLE
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # 当前调度计划
        self.current_schedule: Optional[Schedule] = None
        self.schedule_expiry: Optional[datetime] = None
        
        # 事件队列
        self._grid_events: list = []
        self._price_events: list = []
        
        # 指标
        self.metrics = ControllerMetrics()
        
        # 回调函数
        self._state_callbacks: List[Callable] = []
        self._event_callbacks: List[Callable] = []
        
        # 配置参数
        self.control_interval = self.config.get("control_interval", 60)  # 秒
        self.schedule_horizon = self.config.get("schedule_horizon", 24)  # 小时
        self.emergency_soc_min = self.config.get("emergency_soc_min", 0.05)
        self.emergency_soc_max = self.config.get("emergency_soc_max", 0.95)
        
        logger.info("StorageController initialized")
    
    @classmethod
    def from_config(cls, config_path: str) -> "StorageController":
        """从配置文件创建控制器
        
        Args:
            config_path: 配置文件路径
        
        Returns:
            配置好的控制器实例
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 创建电池模型
        battery_config = config.get("battery", {})
        battery_params = BatteryParams(
            capacity=battery_config.get("capacity", 20.0),
            max_charge_power=battery_config.get("max_charge_power", 10.0),
            max_discharge_power=battery_config.get("max_discharge_power", 10.0),
            efficiency=battery_config.get("efficiency", 0.95),
            min_soc=battery_config.get("min_soc", 0.1),
            max_soc=battery_config.get("max_soc", 0.9),
            degradation_rate=battery_config.get("degradation_rate", 0.02),
        )
        battery = BatteryModel(battery_params, initial_soc=battery_config.get("initial_soc", 0.5))
        
        # 创建电价API
        price_config = config.get("price_api", {})
        price_api = PriceAPI(
            provider=price_config.get("provider", "default"),
            cache_duration=price_config.get("cache_duration", 60),
            region=price_config.get("region", "beijing"),
            api_key=price_config.get("api_key"),
            api_endpoint=price_config.get("api_endpoint"),
        )
        
        # 创建调度器
        scheduler_config = config.get("scheduler", {})
        objective = OptimizationObjective(scheduler_config.get("objective", "balanced"))
        scheduler = EnergyScheduler(battery, price_api, objective)
        scheduler.weights = scheduler_config.get("weights", scheduler.weights)
        
        # 创建控制器
        controller_config = config.get("controller", {})
        return cls(battery, price_api, scheduler, controller_config)
    
    def start(self) -> None:
        """启动控制器"""
        if self._running:
            logger.warning("Controller already running")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._control_loop, daemon=True)
        self._thread.start()
        
        logger.info("StorageController started")
    
    def stop(self) -> None:
        """停止控制器"""
        self._running = False
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        
        self.state = ControllerState.IDLE
        logger.info("StorageController stopped")
    
    def _control_loop(self) -> None:
        """主控制循环"""
        while self._running:
            try:
                self._execute_control_cycle()
            except Exception as e:
                logger.error(f"Error in control loop: {e}", exc_info=True)
                self.state = ControllerState.ERROR
            
            time.sleep(self.control_interval)
    
    def _execute_control_cycle(self) -> None:
        """执行一个控制周期"""
        with self._lock:
            now = datetime.now()
            
            # 1. 检查紧急状态
            if self._check_emergency():
                self._handle_emergency()
                return
            
            # 2. 处理高优先级电网事件
            grid_event = self._get_priority_grid_event()
            if grid_event and grid_event.priority >= 4:
                self._handle_grid_event(grid_event)
                return
            
            # 3. 检查调度计划是否过期
            if self.current_schedule is None or now >= self.schedule_expiry:
                self._update_schedule()
            
            # 4. 执行当前调度计划
            if self.current_schedule:
                self._execute_schedule_point(now)
            
            # 5. 更新指标
            self._update_metrics()
    
    def _check_emergency(self) -> bool:
        """检查是否需要进入紧急状态
        
        Returns:
            是否处于紧急状态
        """
        soc = self.battery.state.soc
        
        # SOC过低或过高
        if soc < self.emergency_soc_min or soc > self.emergency_soc_max:
            return True
        
        # 电池温度过高 (简化处理)
        if self.battery.state.temperature > 50:
            return True
        
        return False
    
    def _handle_emergency(self) -> None:
        """处理紧急状态"""
        soc = self.battery.state.soc
        
        if soc < self.emergency_soc_min:
            # SOC过低，强制充电
            logger.warning(f"Emergency: Low SOC ({soc:.1%}), forcing charge")
            self.state = ControllerState.EMERGENCY
            self.battery.charge(self.battery.params.max_charge_power, 0.25)
            
        elif soc > self.emergency_soc_max:
            # SOC过高，强制放电
            logger.warning(f"Emergency: High SOC ({soc:.1%}), forcing discharge")
            self.state = ControllerState.EMERGENCY
            self.battery.discharge(self.battery.params.max_discharge_power, 0.25)
        
        self._notify_state_change()
    
    def _get_priority_grid_event(self) -> Optional[GridEvent]:
        """获取高优先级电网事件
        
        Returns:
            最高优先级的电网事件，如果没有则返回None
        """
        if not self._grid_events:
            return None
        
        # 按优先级排序
        self._grid_events.sort(key=lambda e: e.priority, reverse=True)
        
        # 检查事件是否过期
        now = datetime.now()
        valid_events = [
            e for e in self._grid_events 
            if now < e.timestamp + timedelta(minutes=e.duration)
        ]
        
        self._grid_events = valid_events
        
        return valid_events[0] if valid_events else None
    
    def _handle_grid_event(self, event: GridEvent) -> None:
        """处理电网事件
        
        Args:
            event: 电网事件
        """
        logger.info(f"Handling grid event: {event.event_type} (priority={event.priority})")
        
        if event.command == "charge":
            power = event.power_limit or self.battery.get_max_charge_power()
            self.battery.charge(power, 0.25)
            self.state = ControllerState.CHARGING
            
        elif event.command == "discharge":
            power = event.power_limit or self.battery.get_max_discharge_power()
            self.battery.discharge(power, 0.25)
            self.state = ControllerState.DISCHARGING
            
        else:  # idle
            self.state = ControllerState.IDLE
        
        self.metrics.grid_events_handled += 1
        self._notify_event("grid", event)
        self._notify_state_change()
    
    def _update_schedule(self) -> None:
        """更新调度计划"""
        logger.info("Updating schedule...")
        
        self.state = ControllerState.OPTIMIZING
        self._notify_state_change()
        
        try:
            # 获取HVAC预测 (简化处理，实际应从HVAC系统获取)
            hvac_forecast = self._get_hvac_forecast()
            
            # 优化调度
            schedule = self.scheduler.optimize(
                horizon=self.schedule_horizon,
                hvac_forecast=hvac_forecast
            )
            
            self.current_schedule = schedule
            self.schedule_expiry = datetime.now() + timedelta(hours=1)
            
            logger.info(f"Schedule updated: {len(schedule.points)} points")
            
        except Exception as e:
            logger.error(f"Failed to update schedule: {e}")
            self.current_schedule = None
    
    def _get_hvac_forecast(self) -> HVACForecast:
        """获取HVAC能耗预测
        
        Returns:
            HVAC预测数据
        """
        # 简化处理：返回默认预测
        # 实际实现应与HVAC系统集成
        now = datetime.now()
        timestamps = [now + timedelta(hours=i) for i in range(self.schedule_horizon)]
        
        # 模拟HVAC需求 (白天高，晚上低)
        power_demands = []
        for t in timestamps:
            hour = t.hour
            if 8 <= hour <= 18:  # 工作时间
                demand = 5.0 + (hour - 12) ** 2 / 20  # 中午需求较低
            else:
                demand = 2.0
            power_demands.append(demand)
        
        return HVACForecast(
            timestamps=timestamps,
            power_demands=power_demands,
            indoor_temps=[24.0] * len(timestamps),
            outdoor_temps=[28.0] * len(timestamps)
        )
    
    def _execute_schedule_point(self, now: datetime) -> None:
        """执行调度计划的当前点
        
        Args:
            now: 当前时间
        """
        point = self.current_schedule.get_point_at(now)
        
        if point is None:
            return
        
        # 执行充放电
        if point.power > 0.1:  # 充电
            self.battery.charge(point.power, 0.25)
            self.state = ControllerState.CHARGING
            
        elif point.power < -0.1:  # 放电
            self.battery.discharge(abs(point.power), 0.25)
            self.state = ControllerState.DISCHARGING
            
        else:  # 待机
            self.battery.update(0, 0.25)
            self.state = ControllerState.IDLE
        
        self._notify_state_change()
    
    def _update_metrics(self) -> None:
        """更新运行指标"""
        # 这里可以添加更多指标计算
        self.metrics.last_update = datetime.now()
    
    def on_price_event(self, event: PriceEvent) -> None:
        """响应电价事件
        
        Args:
            event: 电价事件
        """
        logger.info(f"Received price event: {event.event_type}")
        
        self._price_events.append(event)
        self.metrics.price_events_handled += 1
        
        # 根据事件类型调整策略
        if event.event_type == "price_spike":
            # 电价飙升，尽可能放电
            if self.battery.state.soc > self.battery.params.min_soc + 0.1:
                self.battery.discharge(self.battery.get_max_discharge_power(), 0.25)
                
        elif event.event_type == "price_drop":
            # 电价下降，尽可能充电
            if self.battery.state.soc < self.battery.params.max_soc - 0.1:
                self.battery.charge(self.battery.get_max_charge_power(), 0.25)
        
        # 触发重新调度
        self.schedule_expiry = datetime.now()
        
        self._notify_event("price", event)
    
    def on_grid_event(self, event: GridEvent) -> None:
        """响应电网事件
        
        Args:
            event: 电网事件
        """
        logger.info(f"Received grid event: {event.event_type} (priority={event.priority})")
        
        self._grid_events.append(event)
        
        # 高优先级事件立即处理
        if event.priority >= 4:
            self._handle_grid_event(event)
    
    def get_status(self) -> Dict[str, Any]:
        """获取控制器状态
        
        Returns:
            状态字典
        """
        return {
            "state": self.state.value,
            "battery": {
                "soc": self.battery.state.soc,
                "health": self.battery.state.health,
                "temperature": self.battery.state.temperature,
                "cycle_count": self.battery.state.cycle_count,
            },
            "metrics": {
                "total_cycles": self.metrics.total_cycles,
                "total_energy_charged": self.metrics.total_energy_charged,
                "total_energy_discharged": self.metrics.total_energy_discharged,
                "total_cost_savings": self.metrics.total_cost_savings,
            },
            "current_schedule": {
                "points_count": len(self.current_schedule.points) if self.current_schedule else 0,
                "expiry": self.schedule_expiry.isoformat() if self.schedule_expiry else None,
            }
        }
    
    def register_state_callback(self, callback: Callable[[ControllerState], None]) -> None:
        """注册状态变化回调
        
        Args:
            callback: 回调函数
        """
        self._state_callbacks.append(callback)
    
    def register_event_callback(self, callback: Callable[[str, Any], None]) -> None:
        """注册事件回调
        
        Args:
            callback: 回调函数
        """
        self._event_callbacks.append(callback)
    
    def _notify_state_change(self) -> None:
        """通知状态变化"""
        for callback in self._state_callbacks:
            try:
                callback(self.state)
            except Exception as e:
                logger.error(f"Error in state callback: {e}")
    
    def _notify_event(self, event_type: str, event: Any) -> None:
        """通知事件
        
        Args:
            event_type: 事件类型
            event: 事件对象
        """
        for callback in self._event_callbacks:
            try:
                callback(event_type, event)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")
    
    def run(self) -> None:
        """运行控制器 (阻塞模式)"""
        self.start()
        
        try:
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            self.stop()
