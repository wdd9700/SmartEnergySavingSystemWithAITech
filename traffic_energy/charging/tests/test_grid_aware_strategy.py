#!/usr/bin/env python3
"""
电网感知调度策略测试

测试GridAwareChargingStrategy和GridAwareChargingController的功能。
"""

import unittest
from datetime import datetime, timedelta
import time
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grid_aware_strategy import (
    GridAwareChargingStrategy,
    GridAwareChargingController,
    GridAwareSchedule
)
from grid_calculator import GridPressureCalculator, GridState, GridDataSimulator
from user_schedule import UserScheduleManager, UserSchedule, ScheduleEvent

# 尝试导入scheduler模块
try:
    from scheduler import ChargingScheduler, ChargingRequest, ChargingPile, ChargingSchedule
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    # 创建模拟类
    @dataclass
    class ChargingRequest:
        request_id: str
        vehicle_id: str
        arrival_time: float
        requested_energy: float
        deadline: float
        priority: int = 5
        max_power: float = 50.0
    
    @dataclass
    class ChargingPile:
        pile_id: str
        max_power: float = 50.0
        current_power: float = 0.0
        status: str = "available"
    
    @dataclass
    class ChargingSchedule:
        request_id: str
        pile_id: str
        start_time: float
        end_time: float
        power: float
        energy: float


class MockChargingScheduler:
    """模拟充电调度器"""
    
    def optimize(self, requests, piles):
        """模拟优化"""
        schedules = []
        for i, req in enumerate(requests):
            if i < len(piles):
                pile = piles[i]
                duration = req.requested_energy / req.max_power * 3600
                schedules.append(ChargingSchedule(
                    request_id=req.request_id,
                    pile_id=pile.pile_id,
                    start_time=req.arrival_time,
                    end_time=req.arrival_time + duration,
                    power=req.max_power,
                    energy=req.requested_energy
                ))
        return schedules


class TestGridAwareChargingStrategy(unittest.TestCase):
    """测试GridAwareChargingStrategy类"""
    
    def setUp(self):
        """测试前准备"""
        self.base_scheduler = MockChargingScheduler()
        self.calculator = GridPressureCalculator()
        self.strategy = GridAwareChargingStrategy(self.base_scheduler, self.calculator)
        
        # 创建测试充电桩
        self.piles = [
            ChargingPile(pile_id="pile_1", max_power=50.0),
            ChargingPile(pile_id="pile_2", max_power=50.0)
        ]
    
    def test_normal_grid_schedule(self):
        """测试正常电网状态下的调度"""
        request = ChargingRequest(
            request_id="req_001",
            vehicle_id="vehicle_001",
            arrival_time=time.time(),
            requested_energy=30.0,
            deadline=time.time() + 7200,
            priority=5,
            max_power=50.0
        )
        
        # 正常电网状态
        grid_state = self.calculator.calculate(220.0, 50.0, 0.5)
        self.assertEqual(grid_state.status, "normal")
        
        schedule = self.strategy.schedule(
            request, None, grid_state, self.piles
        )
        
        self.assertEqual(schedule.recommended_action, "normal")
        self.assertEqual(schedule.power_factor, 1.0)
        self.assertFalse(schedule.is_urgent)
    
    def test_warning_grid_schedule(self):
        """测试警告电网状态下的调度"""
        request = ChargingRequest(
            request_id="req_002",
            vehicle_id="vehicle_002",
            arrival_time=time.time(),
            requested_energy=30.0,
            deadline=time.time() + 7200,
            priority=5,
            max_power=50.0
        )
        
        # 警告电网状态
        grid_state = self.calculator.calculate(190.0, 48.5, 0.85)
        self.assertEqual(grid_state.status, "warning")
        
        schedule = self.strategy.schedule(
            request, None, grid_state, self.piles
        )
        
        self.assertEqual(schedule.recommended_action, "reduce_power")
        self.assertEqual(schedule.power_factor, 0.7)
    
    def test_critical_grid_non_urgent_schedule(self):
        """测试紧急电网状态下非紧急请求的调度"""
        request = ChargingRequest(
            request_id="req_003",
            vehicle_id="vehicle_003",
            arrival_time=time.time(),
            requested_energy=30.0,
            deadline=time.time() + 7200,
            priority=5,
            max_power=50.0
        )
        
        # 紧急电网状态
        grid_state = self.calculator.calculate(90.0, 38.0, 1.0)
        self.assertEqual(grid_state.status, "critical")
        
        schedule = self.strategy.schedule(
            request, None, grid_state, self.piles
        )
        
        self.assertEqual(schedule.recommended_action, "postpone")
        self.assertEqual(schedule.power_factor, 0.0)
    
    def test_critical_grid_urgent_schedule(self):
        """测试紧急电网状态下紧急请求的调度"""
        request = ChargingRequest(
            request_id="req_004",
            vehicle_id="vehicle_004",
            arrival_time=time.time(),
            requested_energy=30.0,
            deadline=time.time() + 7200,
            priority=8,
            max_power=50.0
        )
        
        # 创建紧急用户日程
        user_schedule = UserSchedule(
            user_id="user_004",
            vehicle_id="vehicle_004",
            required_soc=0.8,
            required_departure=datetime.now() + timedelta(hours=1),
            flexibility=0.5
        )
        
        # 紧急电网状态
        grid_state = self.calculator.calculate(90.0, 38.0, 1.0)
        
        schedule = self.strategy.schedule(
            request, user_schedule, grid_state, self.piles
        )
        
        # 紧急请求在电网紧急时也应该被满足，但限制功率
        self.assertEqual(schedule.recommended_action, "reduce_power")
        self.assertEqual(schedule.power_factor, 0.3)  # 紧急最小功率因子
        self.assertTrue(schedule.is_urgent)
    
    def test_is_urgent_with_schedule(self):
        """测试紧急需求判断"""
        request = ChargingRequest(
            request_id="req_005",
            vehicle_id="vehicle_005",
            arrival_time=time.time(),
            requested_energy=40.0,  # 需要较多电量
            deadline=time.time() + 3600,  # 1小时后截止
            priority=8,
            max_power=50.0
        )
        
        # 创建紧急用户日程（1小时后需用车）
        user_schedule = UserSchedule(
            user_id="user_005",
            vehicle_id="vehicle_005",
            required_soc=0.8,
            required_departure=datetime.now() + timedelta(hours=1),
            flexibility=0.5
        )
        
        is_urgent = self.strategy._is_urgent(request, user_schedule)
        self.assertTrue(is_urgent)
    
    def test_is_not_urgent_with_schedule(self):
        """测试非紧急需求判断"""
        request = ChargingRequest(
            request_id="req_006",
            vehicle_id="vehicle_006",
            arrival_time=time.time(),
            requested_energy=20.0,
            deadline=time.time() + 14400,  # 4小时后截止
            priority=5,
            max_power=50.0
        )
        
        # 创建非紧急用户日程（4小时后需用车）
        user_schedule = UserSchedule(
            user_id="user_006",
            vehicle_id="vehicle_006",
            required_soc=0.8,
            required_departure=datetime.now() + timedelta(hours=4),
            flexibility=2.0
        )
        
        is_urgent = self.strategy._is_urgent(request, user_schedule)
        self.assertFalse(is_urgent)
    
    def test_batch_schedule_priority(self):
        """测试批量调度优先级"""
        # 创建紧急和非紧急请求
        urgent_request = ChargingRequest(
            request_id="req_urgent",
            vehicle_id="vehicle_urgent",
            arrival_time=time.time(),
            requested_energy=40.0,
            deadline=time.time() + 3600,
            priority=9,
            max_power=50.0
        )
        
        normal_request = ChargingRequest(
            request_id="req_normal",
            vehicle_id="vehicle_normal",
            arrival_time=time.time(),
            requested_energy=20.0,
            deadline=time.time() + 7200,
            priority=5,
            max_power=50.0
        )
        
        # 创建用户日程
        urgent_schedule = UserSchedule(
            user_id="user_urgent",
            vehicle_id="vehicle_urgent",
            required_soc=0.8,
            required_departure=datetime.now() + timedelta(hours=1),
            flexibility=0.5
        )
        
        normal_schedule = UserSchedule(
            user_id="user_normal",
            vehicle_id="vehicle_normal",
            required_soc=0.8,
            required_departure=datetime.now() + timedelta(hours=3),
            flexibility=1.0
        )
        
        user_schedules = {
            "vehicle_urgent": urgent_schedule,
            "vehicle_normal": normal_schedule
        }
        
        requests = [normal_request, urgent_request]
        
        # 警告电网状态
        grid_state = self.calculator.calculate(205.0, 49.5, 0.75)
        
        schedules = self.strategy.schedule_batch(
            requests, user_schedules, grid_state, self.piles
        )
        
        self.assertEqual(len(schedules), 2)
        
        # 验证紧急请求被正确处理
        urgent_result = next(s for s in schedules if s.base_schedule.request_id == "req_urgent")
        self.assertTrue(urgent_result.is_urgent)
    
    def test_grid_aware_schedule_to_dict(self):
        """测试GridAwareSchedule转换为字典"""
        base_schedule = ChargingSchedule(
            request_id="req_001",
            pile_id="pile_1",
            start_time=time.time(),
            end_time=time.time() + 3600,
            power=50.0,
            energy=30.0
        )
        
        grid_state = self.calculator.calculate(220.0, 50.0, 0.5)
        
        schedule = GridAwareSchedule(
            base_schedule=base_schedule,
            grid_state=grid_state,
            is_urgent=False,
            power_factor=0.7,
            recommended_action="reduce_power",
            reason="电网警告状态"
        )
        
        data = schedule.to_dict()
        
        self.assertEqual(data["request_id"], "req_001")
        self.assertEqual(data["power_factor"], 0.7)
        self.assertEqual(data["recommended_action"], "reduce_power")
        self.assertEqual(data["grid_status"], "normal")


class TestGridAwareChargingController(unittest.TestCase):
    """测试GridAwareChargingController类"""
    
    def setUp(self):
        """测试前准备"""
        self.controller = GridAwareChargingController(use_simulator=True)
    
    def tearDown(self):
        """测试后清理"""
        self.controller.stop()
    
    def test_initialization(self):
        """测试控制器初始化"""
        self.assertIsNotNone(self.controller.calculator)
        self.assertIsNotNone(self.controller.schedule_manager)
        self.assertTrue(self.controller._use_simulator)
        self.assertFalse(self.controller.running)
    
    def test_add_user_schedule(self):
        """测试添加用户日程"""
        schedule = UserSchedule(
            user_id="user_001",
            vehicle_id="vehicle_001",
            required_soc=0.8,
            required_departure=datetime.now() + timedelta(hours=2),
            flexibility=1.0
        )
        
        result = self.controller.add_user_schedule("user_001", schedule)
        self.assertTrue(result)
        
        # 验证添加成功
        retrieved = self.controller.schedule_manager.get_schedule("user_001")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.user_id, "user_001")
    
    def test_add_charging_request(self):
        """测试添加充电请求"""
        request = ChargingRequest(
            request_id="req_001",
            vehicle_id="vehicle_001",
            arrival_time=time.time(),
            requested_energy=30.0,
            deadline=time.time() + 7200,
            priority=5,
            max_power=50.0
        )
        
        result = self.controller.add_charging_request(request)
        self.assertTrue(result)
        self.assertEqual(self.controller._pending_requests.qsize(), 1)
    
    def test_fetch_grid_state(self):
        """测试获取电网状态"""
        grid_state = self.controller._fetch_grid_state()
        
        self.assertIsNotNone(grid_state)
        self.assertIn(grid_state.status, ["normal", "warning", "critical"])
        self.assertGreaterEqual(grid_state.pressure_index, 0)
        self.assertLessEqual(grid_state.pressure_index, 1)
    
    def test_get_statistics(self):
        """测试获取统计信息"""
        stats = self.controller.get_statistics()
        
        self.assertIn("controller_status", stats)
        self.assertIn("use_simulator", stats)
        self.assertIn("pending_requests", stats)
        self.assertIn("active_schedules", stats)
        self.assertIn("schedule_summary", stats)
    
    def test_get_schedule_summary_empty(self):
        """测试空调度摘要"""
        summary = self.controller.get_schedule_summary()
        
        self.assertEqual(summary["total"], 0)
        self.assertEqual(summary["urgent"], 0)
        self.assertEqual(summary["normal"], 0)
        self.assertEqual(summary["postponed"], 0)
    
    def test_status_listener(self):
        """测试状态监听器"""
        received_status = []
        
        def listener(status):
            received_status.append(status)
        
        self.controller.add_status_listener(listener)
        
        # 模拟通知
        test_status = {"test": "data"}
        self.controller._notify_status(test_status)
        
        self.assertEqual(len(received_status), 1)
        self.assertEqual(received_status[0], test_status)
        
        # 移除监听器
        self.controller.remove_status_listener(listener)


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_full_workflow(self):
        """测试完整工作流程"""
        # 创建控制器
        controller = GridAwareChargingController(use_simulator=True)
        
        # 添加用户日程
        schedule = UserSchedule(
            user_id="user_int",
            vehicle_id="vehicle_int",
            required_soc=0.8,
            required_departure=datetime.now() + timedelta(hours=2),
            flexibility=1.0
        )
        controller.add_user_schedule("user_int", schedule)
        
        # 添加充电请求
        request = ChargingRequest(
            request_id="req_int",
            vehicle_id="vehicle_int",
            arrival_time=time.time(),
            requested_energy=30.0,
            deadline=time.time() + 7200,
            priority=8,
            max_power=50.0
        )
        controller.add_charging_request(request, user_id="user_int")
        
        # 验证状态
        self.assertEqual(controller._pending_requests.qsize(), 1)
        
        # 获取电网状态
        grid_state = controller._fetch_grid_state()
        self.assertIsNotNone(grid_state)
        
        # 获取统计信息
        stats = controller.get_statistics()
        self.assertEqual(stats["user_schedules"], 1)
        self.assertEqual(stats["pending_requests"], 1)


if __name__ == "__main__":
    unittest.main()
