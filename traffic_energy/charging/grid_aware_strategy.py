#!/usr/bin/env python3
"""
电网感知充电策略模块

实现电网压力感知、用户日程集成和自适应充电调度。

Example:
    >>> from traffic_energy.charging.grid_aware_strategy import GridAwareChargingController
    >>> controller = GridAwareChargingController()
    >>> controller.run()
"""

from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import time
import threading
from queue import Queue

from shared.logger import setup_logger

# 导入依赖模块
try:
    from .scheduler import ChargingScheduler, ChargingRequest, ChargingPile, ChargingSchedule
    from .grid_calculator import GridPressureCalculator, GridState, GridEvent, GridDataSimulator
    from .user_schedule import UserScheduleManager, UserSchedule
    from .grid_monitor import GridMonitor, GridStatus
except ImportError:
    # 直接导入（用于独立测试）
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from charging.scheduler import ChargingScheduler, ChargingRequest, ChargingPile, ChargingSchedule
    from charging.grid_calculator import GridPressureCalculator, GridState, GridEvent, GridDataSimulator
    from charging.user_schedule import UserScheduleManager, UserSchedule
    from charging.grid_monitor import GridMonitor, GridStatus

logger = setup_logger("grid_aware_strategy")


class ChargingAction(Enum):
    """充电操作"""
    START = "start"
    PAUSE = "pause"
    RESUME = "resume"
    STOP = "stop"
    REDUCE_POWER = "reduce_power"
    INCREASE_POWER = "increase_power"


@dataclass
class GridAwareSchedule:
    """电网感知调度方案
    
    扩展基础调度方案，包含电网感知信息。
    
    Attributes:
        base_schedule: 基础调度方案
        grid_state: 电网状态
        user_schedule: 用户日程
        is_urgent: 是否紧急
        power_factor: 功率调整因子
        recommended_action: 推荐操作
        reason: 调度原因说明
    """
    base_schedule: ChargingSchedule
    grid_state: GridState
    user_schedule: Optional[UserSchedule] = None
    is_urgent: bool = False
    power_factor: float = 1.0
    recommended_action: str = "normal"
    reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "request_id": self.base_schedule.request_id,
            "pile_id": self.base_schedule.pile_id,
            "start_time": self.base_schedule.start_time,
            "end_time": self.base_schedule.end_time,
            "power": self.base_schedule.power * self.power_factor,
            "energy": self.base_schedule.energy,
            "grid_status": self.grid_state.status,
            "pressure_index": self.grid_state.pressure_index,
            "is_urgent": self.is_urgent,
            "power_factor": self.power_factor,
            "recommended_action": self.recommended_action,
            "reason": self.reason
        }


class GridAwareChargingStrategy:
    """电网感知充电策略
    
    根据电网状态和用户日程动态调整充电策略。
    
    策略规则:
    1. 电网压力高时: 降低充电功率或暂停充电
    2. 电网压力低时: 提高充电功率
    3. 用户紧急需求: 优先满足，但限制功率
    
    Attributes:
        base_scheduler: 基础调度器
        calculator: 电网压力计算器
        
    Example:
        >>> strategy = GridAwareChargingStrategy(ChargingScheduler())
        >>> schedule = strategy.schedule(request, user_schedule, grid_state, piles)
    """
    
    def __init__(
        self,
        base_scheduler: ChargingScheduler,
        calculator: Optional[GridPressureCalculator] = None
    ) -> None:
        """初始化策略
        
        Args:
            base_scheduler: 基础调度器
            calculator: 电网压力计算器
        """
        self.base_scheduler = base_scheduler
        self.calculator = calculator or GridPressureCalculator()
        
        # 功率调整因子配置
        self.power_factors = {
            "normal": 1.0,      # 正常: 100%功率
            "warning": 0.7,     # 警告: 70%功率
            "critical": 0.5     # 紧急: 50%功率
        }
        
        # 紧急需求功率限制（即使电网紧急也保证最低充电）
        self.urgent_min_power_factor = 0.3
        
        logger.info("初始化电网感知充电策略")
    
    def schedule(
        self,
        request: ChargingRequest,
        user_schedule: Optional[UserSchedule],
        grid_state: GridState,
        piles: List[ChargingPile]
    ) -> GridAwareSchedule:
        """电网感知充电调度
        
        Args:
            request: 充电请求
            user_schedule: 用户日程
            grid_state: 电网状态
            piles: 可用充电桩
            
        Returns:
            电网感知调度方案
        """
        # 基础调度
        base_schedules = self.base_scheduler.optimize([request], piles)
        
        if not base_schedules:
            # 无法调度
            return GridAwareSchedule(
                base_schedule=ChargingSchedule(
                    request_id=request.request_id,
                    pile_id="",
                    start_time=request.arrival_time,
                    end_time=request.arrival_time,
                    power=0.0,
                    energy=0.0
                ),
                grid_state=grid_state,
                user_schedule=user_schedule,
                recommended_action="reject",
                reason="无法安排充电"
            )
        
        base_schedule = base_schedules[0]
        
        # 判断是否为紧急需求
        is_urgent = self._is_urgent(request, user_schedule)
        
        # 根据电网状态调整
        if grid_state.status == "critical":
            return self._handle_critical_grid(
                base_schedule, grid_state, user_schedule, is_urgent
            )
        elif grid_state.status == "warning":
            return self._handle_warning_grid(
                base_schedule, grid_state, user_schedule, is_urgent
            )
        else:
            return self._handle_normal_grid(
                base_schedule, grid_state, user_schedule, is_urgent
            )
    
    def schedule_batch(
        self,
        requests: List[ChargingRequest],
        user_schedules: Dict[str, UserSchedule],
        grid_state: GridState,
        piles: List[ChargingPile]
    ) -> List[GridAwareSchedule]:
        """批量电网感知调度
        
        Args:
            requests: 充电请求列表
            user_schedules: 用户日程字典
            grid_state: 电网状态
            piles: 可用充电桩
            
        Returns:
            电网感知调度方案列表
        """
        # 区分紧急和非紧急请求
        urgent_requests = []
        normal_requests = []
        
        for request in requests:
            user_schedule = user_schedules.get(request.vehicle_id)
            if self._is_urgent(request, user_schedule):
                urgent_requests.append(request)
            else:
                normal_requests.append(request)
        
        schedules = []
        
        # 优先调度紧急请求
        for request in urgent_requests:
            user_schedule = user_schedules.get(request.vehicle_id)
            schedule = self.schedule(request, user_schedule, grid_state, piles)
            schedules.append(schedule)
        
        # 根据电网状态决定是否调度非紧急请求
        if grid_state.status != "critical":
            for request in normal_requests:
                user_schedule = user_schedules.get(request.vehicle_id)
                schedule = self.schedule(request, user_schedule, grid_state, piles)
                schedules.append(schedule)
        else:
            # 电网紧急时，推迟非紧急请求
            for request in normal_requests:
                user_schedule = user_schedules.get(request.vehicle_id)
                schedule = self._postpone_schedule(
                    request, user_schedule, grid_state, delay_hours=2
                )
                schedules.append(schedule)
        
        return schedules
    
    def _is_urgent(
        self,
        request: ChargingRequest,
        schedule: Optional[UserSchedule]
    ) -> bool:
        """判断是否为紧急充电需求
        
        Args:
            request: 充电请求
            schedule: 用户日程
            
        Returns:
            是否紧急
        """
        if not schedule or not schedule.required_departure:
            return False
        
        # 计算所需充电时间（假设电池容量60kWh）
        current_soc = 0.2  # 假设当前20%电量
        energy_needed = (schedule.required_soc - current_soc) * 60
        charge_time_needed = energy_needed / request.max_power  # 小时
        
        # 可用时间
        time_to_departure = (
            schedule.required_departure - datetime.now()
        ).total_seconds() / 3600
        
        # 需用车时间 - 当前时间 < 充电所需时间 + 缓冲(1小时)
        return time_to_departure < charge_time_needed + 1.0
    
    def _handle_critical_grid(
        self,
        base_schedule: ChargingSchedule,
        grid_state: GridState,
        user_schedule: Optional[UserSchedule],
        is_urgent: bool
    ) -> GridAwareSchedule:
        """处理电网紧急状态
        
        Args:
            base_schedule: 基础调度方案
            grid_state: 电网状态
            user_schedule: 用户日程
            is_urgent: 是否紧急
            
        Returns:
            调整后的调度方案
        """
        if is_urgent:
            # 紧急需求: 限制功率但继续充电
            return GridAwareSchedule(
                base_schedule=base_schedule,
                grid_state=grid_state,
                user_schedule=user_schedule,
                is_urgent=True,
                power_factor=self.urgent_min_power_factor,
                recommended_action="reduce_power",
                reason="电网紧急状态，紧急需求限制功率充电"
            )
        else:
            # 非紧急需求: 推迟2小时
            delayed_start = base_schedule.start_time + 2.0 * 3600
            delayed_end = base_schedule.end_time + 2.0 * 3600
            
            postponed_schedule = ChargingSchedule(
                request_id=base_schedule.request_id,
                pile_id="pending",
                start_time=delayed_start,
                end_time=delayed_end,
                power=0.0,
                energy=base_schedule.energy
            )
            
            return GridAwareSchedule(
                base_schedule=postponed_schedule,
                grid_state=grid_state,
                user_schedule=user_schedule,
                is_urgent=False,
                power_factor=0.0,
                recommended_action="postpone",
                reason="电网压力高，推迟2小时充电"
            )
    
    def _handle_warning_grid(
        self,
        base_schedule: ChargingSchedule,
        grid_state: GridState,
        user_schedule: Optional[UserSchedule],
        is_urgent: bool
    ) -> GridAwareSchedule:
        """处理电网警告状态
        
        Args:
            base_schedule: 基础调度方案
            grid_state: 电网状态
            user_schedule: 用户日程
            is_urgent: 是否紧急
            
        Returns:
            调整后的调度方案
        """
        return GridAwareSchedule(
            base_schedule=base_schedule,
            grid_state=grid_state,
            user_schedule=user_schedule,
            is_urgent=is_urgent,
            power_factor=self.power_factors["warning"],
            recommended_action="reduce_power",
            reason="电网警告状态，降低充电功率"
        )
    
    def _handle_normal_grid(
        self,
        base_schedule: ChargingSchedule,
        grid_state: GridState,
        user_schedule: Optional[UserSchedule],
        is_urgent: bool
    ) -> GridAwareSchedule:
        """处理电网正常状态
        
        Args:
            base_schedule: 基础调度方案
            grid_state: 电网状态
            user_schedule: 用户日程
            is_urgent: 是否紧急
            
        Returns:
            调整后的调度方案
        """
        # 电网正常，按计划充电
        return GridAwareSchedule(
            base_schedule=base_schedule,
            grid_state=grid_state,
            user_schedule=user_schedule,
            is_urgent=is_urgent,
            power_factor=self.power_factors["normal"],
            recommended_action="normal",
            reason="电网状态正常，按计划充电"
        )
    
    def _postpone_schedule(
        self,
        request: ChargingRequest,
        user_schedule: Optional[UserSchedule],
        grid_state: GridState,
        delay_hours: float = 2.0
    ) -> GridAwareSchedule:
        """推迟调度方案
        
        Args:
            request: 充电请求
            user_schedule: 用户日程
            grid_state: 电网状态
            delay_hours: 推迟小时数
            
        Returns:
            推迟后的调度方案
        """
        # 创建推迟的调度方案
        delayed_start = request.arrival_time + delay_hours * 3600
        delayed_end = request.deadline + delay_hours * 3600
        
        postponed_schedule = ChargingSchedule(
            request_id=request.request_id,
            pile_id="pending",
            start_time=delayed_start,
            end_time=delayed_end,
            power=0.0,
            energy=request.requested_energy
        )
        
        return GridAwareSchedule(
            base_schedule=postponed_schedule,
            grid_state=grid_state,
            user_schedule=user_schedule,
            is_urgent=False,
            power_factor=0.0,
            recommended_action="postpone",
            reason=f"电网压力高，推迟{delay_hours}小时充电"
        )
    
    def _reduce_power(
        self,
        base_schedule: ChargingSchedule,
        factor: float
    ) -> ChargingSchedule:
        """降低功率
        
        Args:
            base_schedule: 基础调度方案
            factor: 功率因子
            
        Returns:
            调整后的调度方案
        """
        # 延长充电时间以保持充电量
        original_duration = base_schedule.end_time - base_schedule.start_time
        new_power = base_schedule.power * factor
        
        if new_power > 0:
            new_duration = original_duration / factor
            new_end_time = base_schedule.start_time + new_duration
        else:
            new_end_time = base_schedule.end_time
        
        return ChargingSchedule(
            request_id=base_schedule.request_id,
            pile_id=base_schedule.pile_id,
            start_time=base_schedule.start_time,
            end_time=new_end_time,
            power=new_power,
            energy=base_schedule.energy
        )


class GridAwareChargingController:
    """电网感知充电控制器
    
    主控制器，集成电网监测、用户日程管理和充电调度。
    
    Attributes:
        grid_monitor: 电网监测器
        schedule_manager: 用户日程管理器
        strategy: 电网感知策略
        running: 运行状态
        
    Example:
        >>> controller = GridAwareChargingController()
        >>> controller.run()  # 启动控制循环
    """
    
    def __init__(
        self,
        grid_api_endpoint: Optional[str] = None,
        grid_api_key: Optional[str] = None,
        poll_interval: int = 30,
        use_simulator: bool = False
    ) -> None:
        """初始化控制器
        
        Args:
            grid_api_endpoint: 电网API端点
            grid_api_key: 电网API密钥
            poll_interval: 轮询间隔（秒）
            use_simulator: 是否使用模拟数据
        """
        # 初始化电网监测器
        if use_simulator or not grid_api_endpoint:
            self.grid_simulator = GridDataSimulator()
            self.grid_monitor = None
            self._use_simulator = True
        else:
            self.grid_monitor = GridMonitor()  # 使用现有的GridMonitor
            self.grid_simulator = None
            self._use_simulator = False
        
        # 初始化计算器
        self.calculator = GridPressureCalculator()
        
        # 初始化用户日程管理器
        self.schedule_manager = UserScheduleManager()
        
        # 初始化基础调度器
        try:
            self.base_scheduler = ChargingScheduler()
        except ImportError:
            logger.warning("OR-Tools未安装，使用简化调度器")
            self.base_scheduler = None
        
        # 初始化策略
        if self.base_scheduler:
            self.strategy = GridAwareChargingStrategy(
                self.base_scheduler,
                self.calculator
            )
        else:
            self.strategy = None
        
        # 配置
        self.poll_interval = poll_interval
        self.running = False
        self._control_thread: Optional[threading.Thread] = None
        
        # 待处理请求队列
        self._pending_requests: Queue = Queue()
        
        # 当前调度方案
        self._current_schedules: Dict[str, GridAwareSchedule] = {}
        
        # 状态监听器
        self._status_listeners: List[Callable[[Dict[str, Any]], None]] = []
        
        logger.info("初始化电网感知充电控制器")
    
    def add_charging_request(
        self,
        request: ChargingRequest,
        user_id: Optional[str] = None
    ) -> bool:
        """添加充电请求
        
        Args:
            request: 充电请求
            user_id: 用户ID（可选）
            
        Returns:
            是否成功添加
        """
        self._pending_requests.put((request, user_id))
        logger.info(f"添加充电请求: {request.request_id}")
        return True
    
    def add_user_schedule(self, user_id: str, schedule: UserSchedule) -> bool:
        """添加用户日程
        
        Args:
            user_id: 用户ID
            schedule: 用户日程
            
        Returns:
            是否成功
        """
        return self.schedule_manager.add_schedule(user_id, schedule)
    
    def run(self) -> None:
        """启动控制循环"""
        if self.running:
            logger.warning("控制器已在运行")
            return
        
        self.running = True
        self._control_thread = threading.Thread(target=self._control_loop)
        self._control_thread.daemon = True
        self._control_thread.start()
        
        logger.info("电网感知充电控制器已启动")
    
    def stop(self) -> None:
        """停止控制循环"""
        self.running = False
        if self._control_thread:
            self._control_thread.join(timeout=5.0)
        
        logger.info("电网感知充电控制器已停止")
    
    def _control_loop(self) -> None:
        """控制循环"""
        while self.running:
            try:
                # 获取电网状态
                grid_state = self._fetch_grid_state()
                
                # 处理待处理请求
                pending_requests = []
                while not self._pending_requests.empty():
                    try:
                        request, user_id = self._pending_requests.get_nowait()
                        pending_requests.append((request, user_id))
                    except:
                        break
                
                # 获取可用充电桩（简化实现）
                piles = self._get_available_piles()
                
                # 处理每个请求
                for request, user_id in pending_requests:
                    user_schedule = None
                    if user_id:
                        user_schedule = self.schedule_manager.get_schedule(user_id)
                    
                    if self.strategy:
                        schedule = self.strategy.schedule(
                            request, user_schedule, grid_state, piles
                        )
                        self._current_schedules[request.request_id] = schedule
                        
                        # 执行调度
                        self._execute_schedule(schedule)
                
                # 通知状态监听器
                self._notify_status({
                    "timestamp": datetime.now().isoformat(),
                    "grid_state": grid_state.to_dict(),
                    "active_schedules": len(self._current_schedules),
                    "pending_requests": self._pending_requests.qsize()
                })
                
            except Exception as e:
                logger.error(f"控制循环错误: {e}")
            
            # 等待下一次轮询
            time.sleep(self.poll_interval)
    
    def _fetch_grid_state(self) -> GridState:
        """获取电网状态
        
        Returns:
            电网状态
        """
        if self._use_simulator and self.grid_simulator:
            data = self.grid_simulator.generate()
            return self.calculator.calculate(
                voltage=data["voltage"],
                frequency=data["frequency"],
                load_factor=data["load_factor"]
            )
        elif self.grid_monitor:
            # 使用真实电网监测器
            # 这里简化处理，实际应该调用API
            return self.calculator.calculate(
                voltage=220.0,
                frequency=50.0,
                load_factor=0.5
            )
        else:
            # 默认状态
            return GridState(
                timestamp=datetime.now(),
                voltage=220.0,
                frequency=50.0,
                load_factor=0.5,
                status="normal"
            )
    
    def _get_available_piles(self) -> List[ChargingPile]:
        """获取可用充电桩
        
        Returns:
            充电桩列表
        """
        # 简化实现，实际应该从充电桩管理系统获取
        return [
            ChargingPile(pile_id=f"pile_{i}", max_power=50.0, status="available")
            for i in range(4)
        ]
    
    def _execute_schedule(self, schedule: GridAwareSchedule) -> None:
        """执行调度方案
        
        Args:
            schedule: 电网感知调度方案
        """
        action = schedule.recommended_action
        
        if action == "normal":
            logger.info(
                f"执行充电: {schedule.base_schedule.request_id}, "
                f"功率={schedule.base_schedule.power * schedule.power_factor:.1f}kW"
            )
        elif action == "reduce_power":
            logger.info(
                f"降功率充电: {schedule.base_schedule.request_id}, "
                f"功率={schedule.base_schedule.power * schedule.power_factor:.1f}kW, "
                f"原因={schedule.reason}"
            )
        elif action == "postpone":
            logger.info(
                f"推迟充电: {schedule.base_schedule.request_id}, "
                f"新开始时间={schedule.base_schedule.start_time}"
            )
        elif action == "reject":
            logger.warning(f"拒绝充电请求: {schedule.base_schedule.request_id}, 原因={schedule.reason}")
    
    def get_current_schedules(self) -> Dict[str, GridAwareSchedule]:
        """获取当前调度方案
        
        Returns:
            调度方案字典
        """
        return self._current_schedules.copy()
    
    def get_schedule_summary(self) -> Dict[str, Any]:
        """获取调度摘要
        
        Returns:
            摘要信息
        """
        schedules = list(self._current_schedules.values())
        
        if not schedules:
            return {
                "total": 0,
                "urgent": 0,
                "normal": 0,
                "postponed": 0,
                "avg_power_factor": 1.0
            }
        
        return {
            "total": len(schedules),
            "urgent": sum(1 for s in schedules if s.is_urgent),
            "normal": sum(1 for s in schedules if not s.is_urgent and s.recommended_action == "normal"),
            "postponed": sum(1 for s in schedules if s.recommended_action == "postpone"),
            "avg_power_factor": sum(s.power_factor for s in schedules) / len(schedules)
        }
    
    def add_status_listener(
        self,
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """添加状态监听器
        
        Args:
            callback: 回调函数
        """
        self._status_listeners.append(callback)
    
    def remove_status_listener(
        self,
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """移除状态监听器
        
        Args:
            callback: 回调函数
        """
        if callback in self._status_listeners:
            self._status_listeners.remove(callback)
    
    def _notify_status(self, status: Dict[str, Any]) -> None:
        """通知状态
        
        Args:
            status: 状态信息
        """
        for listener in self._status_listeners:
            try:
                listener(status)
            except Exception as e:
                logger.error(f"状态监听器执行失败: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            统计信息
        """
        return {
            "controller_status": "running" if self.running else "stopped",
            "use_simulator": self._use_simulator,
            "poll_interval": self.poll_interval,
            "pending_requests": self._pending_requests.qsize(),
            "active_schedules": len(self._current_schedules),
            "schedule_summary": self.get_schedule_summary(),
            "user_schedules": len(self.schedule_manager._schedules)
        }


if __name__ == "__main__":
    # 简单测试
    controller = GridAwareChargingController(use_simulator=True)
    
    # 添加用户日程
    from datetime import datetime, timedelta
    schedule = UserSchedule(
        user_id="user_001",
        vehicle_id="vehicle_001",
        required_soc=0.8,
        required_departure=datetime.now() + timedelta(hours=2),
        flexibility=1.0
    )
    controller.add_user_schedule("user_001", schedule)
    
    # 添加充电请求
    request = ChargingRequest(
        request_id="req_001",
        vehicle_id="vehicle_001",
        arrival_time=time.time(),
        requested_energy=30.0,
        deadline=time.time() + 7200,
        priority=8,
        max_power=50.0
    )
    controller.add_charging_request(request, user_id="user_001")
    
    # 运行一次控制循环
    grid_state = controller._fetch_grid_state()
    print(f"电网状态: {grid_state}")
    
    print("电网感知充电控制器测试完成")
