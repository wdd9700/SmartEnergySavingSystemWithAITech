"""
调度优化器单元测试
"""

import unittest
from datetime import datetime, timedelta

from building_energy.energy_storage.battery_model import BatteryModel, BatteryParams
from building_energy.energy_storage.price_api import PriceAPI, ElectricityPrice, PriceSchedule
from building_energy.energy_storage.scheduler import (
    EnergyScheduler, Schedule, SchedulePoint, HVACForecast,
    OptimizationObjective
)


class TestSchedulePoint(unittest.TestCase):
    """测试调度点类"""
    
    def test_creation(self):
        """测试创建调度点"""
        point = SchedulePoint(
            timestamp=datetime.now(),
            power=5.0,
            soc=0.6,
            price=0.8
        )
        self.assertEqual(point.power, 5.0)
        self.assertEqual(point.soc, 0.6)
    
    def test_is_charging(self):
        """测试充电状态判断"""
        charging_point = SchedulePoint(
            timestamp=datetime.now(),
            power=5.0,
            soc=0.6,
            price=0.8
        )
        self.assertTrue(charging_point.is_charging)
        self.assertFalse(charging_point.is_discharging)
    
    def test_is_discharging(self):
        """测试放电状态判断"""
        discharging_point = SchedulePoint(
            timestamp=datetime.now(),
            power=-3.0,
            soc=0.5,
            price=1.0
        )
        self.assertFalse(discharging_point.is_charging)
        self.assertTrue(discharging_point.is_discharging)


class TestSchedule(unittest.TestCase):
    """测试调度计划类"""
    
    def setUp(self):
        """测试前准备"""
        now = datetime.now()
        self.points = [
            SchedulePoint(
                timestamp=now + timedelta(hours=i),
                power=5.0 if i < 12 else -3.0,
                soc=0.5 + i * 0.01,
                price=0.6 if i < 8 else 1.0
            )
            for i in range(24)
        ]
        self.schedule = Schedule(
            points=self.points,
            total_cost=100.0,
            total_energy_charged=50.0,
            total_energy_discharged=40.0
        )
    
    def test_get_point_at(self):
        """测试获取指定时间点"""
        target_time = self.points[5].timestamp
        point = self.schedule.get_point_at(target_time)
        self.assertIsNotNone(point)
        self.assertEqual(point.timestamp.hour, target_time.hour)
    
    def test_get_point_at_not_found(self):
        """测试获取不存在的时间点"""
        future_time = datetime.now() + timedelta(days=1)
        point = self.schedule.get_point_at(future_time)
        self.assertIsNone(point)


class TestHVACForecast(unittest.TestCase):
    """测试HVAC预测类"""
    
    def test_get_demand_at(self):
        """测试获取指定小时需求"""
        forecast = HVACForecast(
            timestamps=[datetime.now() + timedelta(hours=i) for i in range(24)],
            power_demands=[5.0] * 24
        )
        demand = forecast.get_demand_at(5)
        self.assertEqual(demand, 5.0)
    
    def test_get_demand_at_out_of_range(self):
        """测试获取超出范围的小时需求"""
        forecast = HVACForecast(
            power_demands=[5.0] * 10
        )
        demand = forecast.get_demand_at(20)
        self.assertEqual(demand, 0.0)


class TestEnergyScheduler(unittest.TestCase):
    """测试调度优化器"""
    
    def setUp(self):
        """测试前准备"""
        # 创建电池模型
        battery_params = BatteryParams(
            capacity=20.0,
            max_charge_power=10.0,
            max_discharge_power=10.0,
            efficiency=0.95,
            min_soc=0.1,
            max_soc=0.9
        )
        self.battery = BatteryModel(battery_params, initial_soc=0.5)
        
        # 创建电价API
        self.price_api = PriceAPI(provider="default")
        
        # 创建调度器
        self.scheduler = EnergyScheduler(
            self.battery,
            self.price_api,
            objective=OptimizationObjective.BALANCED
        )
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.scheduler.battery, self.battery)
        self.assertEqual(self.scheduler.price_api, self.price_api)
        self.assertEqual(self.scheduler.objective, OptimizationObjective.BALANCED)
    
    def test_optimize_basic(self):
        """测试基本优化"""
        schedule = self.scheduler.optimize(horizon=24)
        
        # 验证调度计划
        self.assertIsInstance(schedule, Schedule)
        self.assertEqual(len(schedule.points), 24)
    
    def test_optimize_with_hvac(self):
        """测试带HVAC预测的优化"""
        hvac_forecast = HVACForecast(
            timestamps=[datetime.now() + timedelta(hours=i) for i in range(24)],
            power_demands=[5.0] * 8 + [8.0] * 8 + [5.0] * 8  # 白天需求高
        )
        
        schedule = self.scheduler.optimize(
            horizon=24,
            hvac_forecast=hvac_forecast
        )
        
        self.assertEqual(len(schedule.points), 24)
    
    def test_optimize_peak_shaving(self):
        """测试削峰优化"""
        schedule = self.scheduler.optimize_peak_shaving(
            peak_threshold=5.0,
            horizon=24
        )
        
        self.assertIsInstance(schedule, Schedule)
    
    def test_calculate_savings(self):
        """测试节省计算"""
        schedule = self.scheduler.optimize(horizon=24)
        
        savings = self.scheduler.calculate_savings(schedule)
        
        self.assertIn("baseline_cost", savings)
        self.assertIn("actual_cost", savings)
        self.assertIn("savings", savings)
        self.assertIn("savings_percent", savings)
    
    def test_calculate_savings_with_baseline(self):
        """测试带基准成本的节省计算"""
        schedule = self.scheduler.optimize(horizon=24)
        
        baseline_cost = 200.0
        savings = self.scheduler.calculate_savings(schedule, baseline_cost)
        
        self.assertEqual(savings["baseline_cost"], baseline_cost)


class TestOptimizationStrategies(unittest.TestCase):
    """测试不同优化策略"""
    
    def setUp(self):
        """测试前准备"""
        battery_params = BatteryParams(
            capacity=20.0,
            max_charge_power=10.0,
            max_discharge_power=10.0,
            efficiency=0.95
        )
        self.battery = BatteryModel(battery_params, initial_soc=0.5)
        self.price_api = PriceAPI(provider="default")
    
    def test_cost_objective(self):
        """测试成本优化目标"""
        scheduler = EnergyScheduler(
            self.battery,
            self.price_api,
            objective=OptimizationObjective.COST
        )
        
        schedule = scheduler.optimize(horizon=24)
        self.assertIsNotNone(schedule)
    
    def test_comfort_objective(self):
        """测试舒适度优化目标"""
        scheduler = EnergyScheduler(
            self.battery,
            self.price_api,
            objective=OptimizationObjective.COMFORT
        )
        
        schedule = scheduler.optimize(horizon=24)
        self.assertIsNotNone(schedule)
    
    def test_grid_objective(self):
        """测试电网优化目标"""
        scheduler = EnergyScheduler(
            self.battery,
            self.price_api,
            objective=OptimizationObjective.GRID
        )
        
        schedule = scheduler.optimize(horizon=24)
        self.assertIsNotNone(schedule)


class TestScheduleConstraints(unittest.TestCase):
    """测试调度约束"""
    
    def setUp(self):
        """测试前准备"""
        battery_params = BatteryParams(
            capacity=10.0,
            max_charge_power=5.0,
            max_discharge_power=5.0,
            efficiency=0.95,
            min_soc=0.1,
            max_soc=0.9
        )
        self.battery = BatteryModel(battery_params, initial_soc=0.5)
        self.price_api = PriceAPI(provider="default")
        self.scheduler = EnergyScheduler(self.battery, self.price_api)
    
    def test_soc_constraints_respected(self):
        """测试SOC约束被遵守"""
        schedule = self.scheduler.optimize(horizon=24)
        
        for point in schedule.points:
            # SOC应在允许范围内
            self.assertGreaterEqual(point.soc, 0.1 - 0.01)  # 允许小误差
            self.assertLessEqual(point.soc, 0.9 + 0.01)
    
    def test_power_constraints_respected(self):
        """测试功率约束被遵守"""
        schedule = self.scheduler.optimize(horizon=24)
        
        for point in schedule.points:
            # 功率应在允许范围内
            self.assertGreaterEqual(point.power, -5.0 - 0.1)
            self.assertLessEqual(point.power, 5.0 + 0.1)


class TestFallbackScheduler(unittest.TestCase):
    """测试降级调度器"""
    
    def setUp(self):
        """测试前准备"""
        battery_params = BatteryParams(capacity=10.0)
        self.battery = BatteryModel(battery_params)
        self.price_api = PriceAPI(provider="default")
        self.scheduler = EnergyScheduler(self.battery, self.price_api)
    
    def test_fallback_schedule_creation(self):
        """测试降级计划创建"""
        schedule = self.scheduler._create_fallback_schedule(horizon=12)
        
        self.assertEqual(len(schedule.points), 12)
        
        # 降级计划应保持SOC不变
        for point in schedule.points:
            self.assertEqual(point.power, 0.0)


if __name__ == "__main__":
    unittest.main()
