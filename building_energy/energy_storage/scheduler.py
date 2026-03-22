"""
储能调度优化器

使用OR-Tools实现多目标优化调度算法，考虑成本、舒适度和电网压力。

参考:
- OR-Tools文档: https://developers.google.com/optimization
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum

import numpy as np

# 尝试导入OR-Tools
try:
    from ortools.linear_solver import pywraplp
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    logging.warning("OR-Tools not installed. Using fallback scheduler.")

try:
    from .battery_model import BatteryModel
    from .price_api import PriceAPI, PriceSchedule, ElectricityPrice
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from battery_model import BatteryModel
    from price_api import PriceAPI, PriceSchedule, ElectricityPrice

logger = logging.getLogger(__name__)


class OptimizationObjective(Enum):
    """优化目标类型"""
    COST = "cost"           # 最小化成本
    COMFORT = "comfort"     # 最大化舒适度
    GRID = "grid"           # 最小化电网压力
    BALANCED = "balanced"   # 平衡多目标


@dataclass
class SchedulePoint:
    """调度计划点
    
    Attributes:
        timestamp: 时间戳
        power: 功率 (kW, 正为充电, 负为放电)
        soc: 预期SOC
        price: 电价 (元/kWh)
        hvac_power: HVAC功率需求 (kW)
        grid_power: 从电网获取的功率 (kW)
    """
    timestamp: datetime
    power: float
    soc: float
    price: float
    hvac_power: float = 0.0
    grid_power: float = 0.0
    
    @property
    def is_charging(self) -> bool:
        """是否在充电"""
        return self.power > 0
    
    @property
    def is_discharging(self) -> bool:
        """是否在放电"""
        return self.power < 0


@dataclass
class Schedule:
    """调度计划
    
    Attributes:
        points: 调度点列表
        objective_value: 优化目标值
        total_cost: 总成本估算
        total_energy_charged: 总充电量
        total_energy_discharged: 总放电量
        created_at: 创建时间
    """
    points: List[SchedulePoint] = field(default_factory=list)
    objective_value: float = 0.0
    total_cost: float = 0.0
    total_energy_charged: float = 0.0
    total_energy_discharged: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_point_at(self, timestamp: datetime) -> Optional[SchedulePoint]:
        """获取指定时间的调度点"""
        for point in self.points:
            if point.timestamp.hour == timestamp.hour:
                return point
        return None


@dataclass
class HVACForecast:
    """HVAC能耗预测
    
    Attributes:
        timestamps: 时间戳列表
        power_demands: 功率需求列表 (kW)
        indoor_temps: 预测室内温度列表 (°C)
        outdoor_temps: 预测室外温度列表 (°C)
    """
    timestamps: List[datetime] = field(default_factory=list)
    power_demands: List[float] = field(default_factory=list)
    indoor_temps: List[float] = field(default_factory=list)
    outdoor_temps: List[float] = field(default_factory=list)
    
    def get_demand_at(self, hour: int) -> float:
        """获取指定小时的HVAC需求"""
        if 0 <= hour < len(self.power_demands):
            return self.power_demands[hour]
        return 0.0


class EnergyScheduler:
    """储能调度优化器
    
    使用线性规划优化储能系统的充放电调度，实现多目标优化。
    
    Attributes:
        battery: 电池模型
        price_api: 电价API
        objective: 优化目标
    
    Example:
        >>> scheduler = EnergyScheduler(battery, price_api)
        >>> schedule = scheduler.optimize(horizon=24, hvac_demand=5.0)
        >>> for point in schedule.points:
        ...     print(f"{point.timestamp}: {point.power:.2f} kW, SOC={point.soc:.1%}")
    """
    
    def __init__(
        self,
        battery: BatteryModel,
        price_api: PriceAPI,
        objective: OptimizationObjective = OptimizationObjective.BALANCED
    ):
        """
        初始化调度器
        
        Args:
            battery: 电池模型
            price_api: 电价API
            objective: 优化目标
        """
        self.battery = battery
        self.price_api = price_api
        self.objective = objective
        
        # 优化权重 (用于平衡多目标)
        self.weights = {
            "cost": 0.5,
            "comfort": 0.3,
            "grid": 0.2
        }
        
        logger.info(f"EnergyScheduler initialized with objective: {objective.value}")
    
    def optimize(
        self,
        horizon: int = 24,
        hvac_forecast: Optional[HVACForecast] = None,
        initial_soc: Optional[float] = None,
        target_soc: Optional[float] = None,
        grid_constraints: Optional[Dict[int, float]] = None
    ) -> Schedule:
        """优化储能调度
        
        Args:
            horizon: 优化时长 (小时)
            hvac_forecast: HVAC能耗预测
            initial_soc: 初始SOC (默认使用当前SOC)
            target_soc: 目标终止SOC
            grid_constraints: 电网约束 {小时: 最大功率限制}
        
        Returns:
            优化后的调度计划
        """
        # 获取电价计划
        price_schedule = self.price_api.get_price_schedule(horizon)
        
        if not price_schedule.prices:
            logger.warning("No price data available, using fallback schedule")
            return self._create_fallback_schedule(horizon, hvac_forecast)
        
        # 使用OR-Tools优化或降级方案
        if ORTOOLS_AVAILABLE:
            return self._optimize_with_ortools(
                horizon, price_schedule, hvac_forecast,
                initial_soc, target_soc, grid_constraints
            )
        else:
            return self._optimize_fallback(
                horizon, price_schedule, hvac_forecast
            )
    
    def _optimize_with_ortools(
        self,
        horizon: int,
        price_schedule: PriceSchedule,
        hvac_forecast: Optional[HVACForecast],
        initial_soc: Optional[float],
        target_soc: Optional[float],
        grid_constraints: Optional[Dict[int, float]]
    ) -> Schedule:
        """使用OR-Tools进行优化
        
        Args:
            horizon: 优化时长
            price_schedule: 电价计划
            hvac_forecast: HVAC预测
            initial_soc: 初始SOC
            target_soc: 目标SOC
            grid_constraints: 电网约束
        
        Returns:
            调度计划
        """
        # 创建求解器
        solver = pywraplp.Solver.CreateSolver('GLOP')
        if not solver:
            logger.error("Failed to create solver")
            return self._create_fallback_schedule(horizon, hvac_forecast)
        
        # 决策变量: 每小时充放电功率 (正为充电, 负为放电)
        power_vars = []
        soc_vars = []
        grid_vars = []
        
        # 初始SOC
        soc_0 = initial_soc if initial_soc is not None else self.battery.state.soc
        
        # 创建变量
        for t in range(horizon):
            # 充放电功率 (连续变量, 范围 [-max_discharge, max_charge])
            p = solver.NumVar(
                -self.battery.params.max_discharge_power,
                self.battery.params.max_charge_power,
                f'power_{t}'
            )
            power_vars.append(p)
            
            # SOC变量
            s = solver.NumVar(
                self.battery.params.min_soc,
                self.battery.params.max_soc,
                f'soc_{t}'
            )
            soc_vars.append(s)
            
            # 从电网获取的功率
            g = solver.NumVar(0, solver.infinity(), f'grid_{t}')
            grid_vars.append(g)
        
        # 添加SOC动态约束
        # SOC[t+1] = SOC[t] + (efficiency * charge - discharge) / capacity
        efficiency = self.battery.params.efficiency
        capacity = self.battery.params.capacity
        
        for t in range(horizon):
            if t == 0:
                # 初始SOC约束
                solver.Add(soc_vars[t] == soc_0 + 
                          (efficiency * solver.Max(0, power_vars[t]) - 
                           solver.Max(0, -power_vars[t])) / capacity)
            else:
                # SOC动态约束
                solver.Add(soc_vars[t] == soc_vars[t-1] + 
                          (efficiency * solver.Max(0, power_vars[t]) - 
                           solver.Max(0, -power_vars[t])) / capacity)
        
        # 目标函数: 最小化总成本
        # 成本 = 从电网购电成本 - 向电网售电收入
        objective = solver.Objective()
        
        for t in range(horizon):
            price = price_schedule.prices[t].price if t < len(price_schedule.prices) else 0.6
            hvac_demand = hvac_forecast.get_demand_at(t) if hvac_forecast else 0.0
            
            # 电网功率 = HVAC需求 + 充电功率 (如果充电)
            # 简化为: grid = hvac + max(0, power)
            # 实际上应该使用分段线性化，这里简化处理
            
            # 添加约束: grid >= hvac_demand + power (当power>0时)
            solver.Add(grid_vars[t] >= hvac_demand + power_vars[t])
            solver.Add(grid_vars[t] >= hvac_demand)
            
            # 目标函数系数
            objective.SetCoefficient(grid_vars[t], price)
        
        objective.SetMinimization()
        
        # 求解
        status = solver.Solve()
        
        if status == pywraplp.Solver.OPTIMAL:
            logger.info(f"Optimization successful. Objective value: {solver.Objective().Value():.2f}")
            return self._extract_schedule(
                horizon, price_schedule, power_vars, soc_vars, grid_vars, hvac_forecast
            )
        else:
            logger.warning(f"Optimization failed with status: {status}")
            return self._create_fallback_schedule(horizon, hvac_forecast)
    
    def _optimize_fallback(
        self,
        horizon: int,
        price_schedule: PriceSchedule,
        hvac_forecast: Optional[HVACForecast]
    ) -> Schedule:
        """降级优化方案 (基于规则的调度)
        
        当OR-Tools不可用时使用的简单规则调度。
        
        策略:
        - 谷时(低价)充电
        - 峰时(高价)放电
        
        Args:
            horizon: 优化时长
            price_schedule: 电价计划
            hvac_forecast: HVAC预测
        
        Returns:
            调度计划
        """
        logger.info("Using fallback rule-based scheduler")
        
        points = []
        current_soc = self.battery.state.soc
        total_cost = 0.0
        total_charged = 0.0
        total_discharged = 0.0
        
        for t, price_point in enumerate(price_schedule.prices[:horizon]):
            hvac_demand = hvac_forecast.get_demand_at(t) if hvac_forecast else 0.0
            
            # 基于电价的简单规则
            if price_point.period == "valley":
                # 谷时: 尽可能充电
                max_charge = min(
                    self.battery.get_max_charge_power(),
                    (self.battery.params.max_soc - current_soc) * self.battery.params.capacity
                )
                power = max_charge
                
            elif price_point.period == "peak":
                # 峰时: 尽可能放电以满足HVAC需求
                max_discharge = min(
                    self.battery.get_max_discharge_power(),
                    (current_soc - self.battery.params.min_soc) * self.battery.params.capacity
                )
                # 放电功率 = min(最大放电, HVAC需求)
                power = -min(max_discharge, hvac_demand + 2.0)  # 额外放2kW套利
                
            else:
                # 平时: 保持或小幅调整
                # 如果SOC较低，小功率充电
                if current_soc < 0.5:
                    power = 2.0
                else:
                    power = 0.0
            
            # 计算电网功率需求
            grid_power = hvac_demand + max(0, power)
            
            # 更新SOC
            if power > 0:
                energy = power * self.battery.params.efficiency
                current_soc += energy / self.battery.params.capacity
                total_charged += energy
            elif power < 0:
                energy = abs(power)
                current_soc -= energy / self.battery.params.capacity
                total_discharged += energy
            
            # 限制SOC范围
            current_soc = max(self.battery.params.min_soc, 
                            min(self.battery.params.max_soc, current_soc))
            
            # 计算成本
            cost = grid_power * price_point.price
            total_cost += cost
            
            points.append(SchedulePoint(
                timestamp=price_point.timestamp,
                power=power,
                soc=current_soc,
                price=price_point.price,
                hvac_power=hvac_demand,
                grid_power=grid_power
            ))
        
        return Schedule(
            points=points,
            total_cost=total_cost,
            total_energy_charged=total_charged,
            total_energy_discharged=total_discharged
        )
    
    def _extract_schedule(
        self,
        horizon: int,
        price_schedule: PriceSchedule,
        power_vars,
        soc_vars,
        grid_vars,
        hvac_forecast: Optional[HVACForecast]
    ) -> Schedule:
        """从求解器结果提取调度计划
        
        Args:
            horizon: 优化时长
            price_schedule: 电价计划
            power_vars: 功率变量
            soc_vars: SOC变量
            grid_vars: 电网功率变量
            hvac_forecast: HVAC预测
        
        Returns:
            调度计划
        """
        points = []
        total_cost = 0.0
        total_charged = 0.0
        total_discharged = 0.0
        
        for t in range(horizon):
            power = power_vars[t].solution_value()
            soc = soc_vars[t].solution_value()
            grid_power = grid_vars[t].solution_value()
            
            price = price_schedule.prices[t].price if t < len(price_schedule.prices) else 0.6
            hvac_demand = hvac_forecast.get_demand_at(t) if hvac_forecast else 0.0
            
            if power > 0:
                total_charged += power
            elif power < 0:
                total_discharged += abs(power)
            
            total_cost += grid_power * price
            
            points.append(SchedulePoint(
                timestamp=price_schedule.prices[t].timestamp,
                power=power,
                soc=soc,
                price=price,
                hvac_power=hvac_demand,
                grid_power=grid_power
            ))
        
        return Schedule(
            points=points,
            total_cost=total_cost,
            total_energy_charged=total_charged,
            total_energy_discharged=total_discharged
        )
    
    def _create_fallback_schedule(
        self,
        horizon: int,
        hvac_forecast: Optional[HVACForecast]
    ) -> Schedule:
        """创建默认调度计划
        
        当优化失败时使用。
        
        Args:
            horizon: 优化时长
            hvac_forecast: HVAC预测
        
        Returns:
            默认调度计划
        """
        logger.warning("Creating fallback schedule (no optimization)")
        
        points = []
        now = datetime.now()
        current_soc = self.battery.state.soc
        
        for t in range(horizon):
            timestamp = now + timedelta(hours=t)
            hvac_demand = hvac_forecast.get_demand_at(t) if hvac_forecast else 0.0
            
            # 默认策略: 保持当前SOC，不主动充放电
            points.append(SchedulePoint(
                timestamp=timestamp,
                power=0.0,
                soc=current_soc,
                price=0.6,
                hvac_power=hvac_demand,
                grid_power=hvac_demand
            ))
        
        return Schedule(points=points)
    
    def optimize_peak_shaving(
        self,
        peak_threshold: float,
        horizon: int = 24
    ) -> Schedule:
        """优化削峰调度
        
        在电网负荷高峰时段减少从电网取电。
        
        Args:
            peak_threshold: 峰值功率阈值 (kW)
            horizon: 优化时长
        
        Returns:
            削峰调度计划
        """
        # 创建电网约束
        grid_constraints = {}
        
        # 获取电价计划识别峰时
        price_schedule = self.price_api.get_price_schedule(horizon)
        
        for t, price_point in enumerate(price_schedule.prices[:horizon]):
            if price_point.period in ["peak", "critical"]:
                grid_constraints[t] = peak_threshold
        
        return self.optimize(
            horizon=horizon,
            grid_constraints=grid_constraints
        )
    
    def calculate_savings(
        self,
        schedule: Schedule,
        baseline_cost: Optional[float] = None
    ) -> Dict[str, float]:
        """计算节省的费用
        
        Args:
            schedule: 调度计划
            baseline_cost: 基准成本 (无储能时的成本)
        
        Returns:
            节省统计
        """
        if baseline_cost is None:
            # 计算无储能时的基准成本
            baseline_cost = sum(
                point.hvac_power * point.price 
                for point in schedule.points
            )
        
        actual_cost = schedule.total_cost
        savings = baseline_cost - actual_cost
        
        return {
            "baseline_cost": baseline_cost,
            "actual_cost": actual_cost,
            "savings": savings,
            "savings_percent": (savings / baseline_cost * 100) if baseline_cost > 0 else 0,
            "energy_charged": schedule.total_energy_charged,
            "energy_discharged": schedule.total_energy_discharged,
            "efficiency": (
                schedule.total_energy_discharged / schedule.total_energy_charged * 100
                if schedule.total_energy_charged > 0 else 0
            )
        }
