"""
电池模型单元测试
"""

import unittest
from datetime import datetime

from building_energy.energy_storage.battery_model import (
    BatteryParams, BatteryState, BatteryModel
)


class TestBatteryParams(unittest.TestCase):
    """测试电池参数类"""
    
    def test_valid_params(self):
        """测试有效参数"""
        params = BatteryParams(
            capacity=20.0,
            max_charge_power=10.0,
            max_discharge_power=10.0,
            efficiency=0.95,
            min_soc=0.1,
            max_soc=0.9
        )
        self.assertEqual(params.capacity, 20.0)
        self.assertEqual(params.efficiency, 0.95)
    
    def test_invalid_capacity(self):
        """测试无效容量"""
        with self.assertRaises(ValueError):
            BatteryParams(capacity=200.0)  # 超出范围
        
        with self.assertRaises(ValueError):
            BatteryParams(capacity=1.0)  # 低于范围
    
    def test_invalid_efficiency(self):
        """测试无效效率"""
        with self.assertRaises(ValueError):
            BatteryParams(efficiency=1.5)  # 大于1
        
        with self.assertRaises(ValueError):
            BatteryParams(efficiency=0)  # 等于0
    
    def test_invalid_soc_range(self):
        """测试无效SOC范围"""
        with self.assertRaises(ValueError):
            BatteryParams(min_soc=0.5, max_soc=0.3)  # min > max
        
        with self.assertRaises(ValueError):
            BatteryParams(min_soc=-0.1)  # 负数
        
        with self.assertRaises(ValueError):
            BatteryParams(max_soc=1.5)  # 大于1


class TestBatteryState(unittest.TestCase):
    """测试电池状态类"""
    
    def test_valid_state(self):
        """测试有效状态"""
        state = BatteryState(soc=0.5, health=1.0)
        self.assertEqual(state.soc, 0.5)
        self.assertEqual(state.health, 1.0)
    
    def test_invalid_soc(self):
        """测试无效SOC"""
        with self.assertRaises(ValueError):
            BatteryState(soc=1.5)
        
        with self.assertRaises(ValueError):
            BatteryState(soc=-0.1)
    
    def test_invalid_health(self):
        """测试无效健康度"""
        with self.assertRaises(ValueError):
            BatteryState(health=1.5)


class TestBatteryModel(unittest.TestCase):
    """测试电池模型类"""
    
    def setUp(self):
        """测试前准备"""
        self.params = BatteryParams(
            capacity=10.0,
            max_charge_power=5.0,
            max_discharge_power=5.0,
            efficiency=0.95,
            min_soc=0.1,
            max_soc=0.9
        )
        self.battery = BatteryModel(self.params, initial_soc=0.5)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.battery.state.soc, 0.5)
        self.assertEqual(self.battery.params.capacity, 10.0)
    
    def test_charge(self):
        """测试充电"""
        initial_soc = self.battery.state.soc
        energy = self.battery.charge(5.0, 1.0)  # 5kW充电1小时
        
        # 验证充入了电量
        self.assertGreater(energy, 0)
        # 验证SOC上升
        self.assertGreater(self.battery.state.soc, initial_soc)
    
    def test_charge_limits(self):
        """测试充电限制"""
        # 设置SOC接近上限
        self.battery.state.soc = 0.85
        
        # 尝试大量充电
        energy = self.battery.charge(10.0, 1.0)
        
        # SOC不应超过max_soc
        self.assertLessEqual(self.battery.state.soc, self.params.max_soc + 0.01)
    
    def test_discharge(self):
        """测试放电"""
        initial_soc = self.battery.state.soc
        energy = self.battery.discharge(3.0, 1.0)  # 3kW放电1小时
        
        # 验证放出了电量
        self.assertGreater(energy, 0)
        # 验证SOC下降
        self.assertLess(self.battery.state.soc, initial_soc)
    
    def test_discharge_limits(self):
        """测试放电限制"""
        # 设置SOC接近下限
        self.battery.state.soc = 0.15
        
        # 尝试大量放电
        energy = self.battery.discharge(10.0, 1.0)
        
        # SOC不应低于min_soc
        self.assertGreaterEqual(self.battery.state.soc, self.params.min_soc - 0.01)
    
    def test_invalid_charge_power(self):
        """测试无效充电功率"""
        with self.assertRaises(ValueError):
            self.battery.charge(-5.0, 1.0)  # 负功率
    
    def test_invalid_discharge_power(self):
        """测试无效放电功率"""
        with self.assertRaises(ValueError):
            self.battery.discharge(-3.0, 1.0)  # 负功率
    
    def test_temperature_factor(self):
        """测试温度影响"""
        # 设置正常温度
        self.battery.set_temperature(25.0)
        factor_normal = self.battery._get_temperature_factor()
        self.assertAlmostEqual(factor_normal, 1.0, places=2)
        
        # 设置极端温度
        self.battery.set_temperature(50.0)
        factor_hot = self.battery._get_temperature_factor()
        self.assertLess(factor_hot, 1.0)
    
    def test_cycle_count(self):
        """测试循环计数"""
        initial_cycles = self.battery.state.cycle_count
        
        # 充放电一个完整循环
        self.battery.charge(5.0, 2.0)  # 充入约10kWh
        self.battery.discharge(5.0, 2.0)  # 放出约10kWh
        
        # 循环次数应增加
        self.assertGreaterEqual(self.battery.state.cycle_count, initial_cycles)
    
    def test_health_degradation(self):
        """测试健康度衰减"""
        initial_health = self.battery.state.health
        
        # 模拟多次循环
        for _ in range(100):
            self.battery.charge(5.0, 0.1)
            self.battery.discharge(5.0, 0.1)
        
        # 健康度应下降
        self.assertLessEqual(self.battery.state.health, initial_health)
        self.assertGreaterEqual(self.battery.state.health, 0.6)  # 不低于最低值
    
    def test_get_available_energy(self):
        """测试获取可用电量"""
        available = self.battery.get_available_energy()
        expected = (0.5 - 0.1) * 10.0  # (SOC - min_soc) * capacity
        self.assertAlmostEqual(available, expected, places=1)
    
    def test_get_available_capacity(self):
        """测试获取可用容量"""
        available = self.battery.get_available_capacity()
        expected = (0.9 - 0.5) * 10.0  # (max_soc - SOC) * capacity
        self.assertAlmostEqual(available, expected, places=1)
    
    def test_get_max_powers(self):
        """测试获取最大功率"""
        # 正常SOC下
        max_charge = self.battery.get_max_charge_power()
        max_discharge = self.battery.get_max_discharge_power()
        self.assertGreater(max_charge, 0)
        self.assertGreater(max_discharge, 0)
        
        # SOC达到上限时
        self.battery.state.soc = 0.9
        max_charge = self.battery.get_max_charge_power()
        self.assertEqual(max_charge, 0.0)
        
        # SOC达到下限时
        self.battery.state.soc = 0.1
        max_discharge = self.battery.get_max_discharge_power()
        self.assertEqual(max_discharge, 0.0)
    
    def test_reset(self):
        """测试重置"""
        # 进行一些操作
        self.battery.charge(5.0, 1.0)
        self.battery.state.cycle_count = 10
        
        # 重置
        self.battery.reset(soc=0.6)
        
        self.assertEqual(self.battery.state.soc, 0.6)
        self.assertEqual(self.battery.state.cycle_count, 0)
    
    def test_to_dict(self):
        """测试转换为字典"""
        data = self.battery.to_dict()
        
        self.assertIn("params", data)
        self.assertIn("state", data)
        self.assertEqual(data["params"]["capacity"], 10.0)
        self.assertEqual(data["state"]["soc"], 0.5)


class TestBatteryEfficiency(unittest.TestCase):
    """测试电池效率"""
    
    def test_round_trip_efficiency(self):
        """测试往返效率"""
        params = BatteryParams(
            capacity=10.0,
            max_charge_power=5.0,
            max_discharge_power=5.0,
            efficiency=0.95,
            min_soc=0.2,  # 设置较高下限以确保有足够容量
            max_soc=0.8
        )
        battery = BatteryModel(params, initial_soc=0.5)
        
        # 记录初始SOC
        initial_soc = battery.state.soc
        
        # 充电
        charged_energy = battery.charge(3.0, 1.0)
        soc_after_charge = battery.state.soc
        
        # 放电相同能量
        discharged_energy = battery.discharge(3.0, 1.0)
        soc_after_discharge = battery.state.soc
        
        # 由于效率损失，放电能量应小于充电能量
        self.assertLess(discharged_energy, charged_energy)
        
        # SOC应回到接近初始值 (考虑效率损失)
        self.assertLess(battery.state.soc, soc_after_charge)


if __name__ == "__main__":
    unittest.main()
