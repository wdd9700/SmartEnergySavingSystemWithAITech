#!/usr/bin/env python3
"""
用户日程管理模块

管理用户日程、充电需求预测和日程变更处理。

Example:
    >>> from traffic_energy.charging.user_schedule import UserScheduleManager, UserSchedule
    >>> manager = UserScheduleManager()
    >>> schedule = UserSchedule(
    ...     user_id="user_001",
    ...     vehicle_id="vehicle_001",
    ...     required_soc=0.8,
    ...     required_departure=datetime.now() + timedelta(hours=2)
    ... )
    >>> manager.add_schedule("user_001", schedule)
"""

from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import time
import json
import threading

from shared.logger import setup_logger

logger = setup_logger("user_schedule")


class SchedulePriority(Enum):
    """日程优先级"""
    LOW = 1
    MEDIUM = 5
    HIGH = 10


@dataclass
class ScheduleEvent:
    """日程事件
    
    Attributes:
        event_id: 事件ID
        start_time: 开始时间
        end_time: 结束时间
        location: 地点
        priority: 优先级 (1-10)
        requires_vehicle: 是否需要用车
        description: 事件描述
    """
    event_id: str
    start_time: datetime
    end_time: datetime
    location: str
    priority: int = 5
    requires_vehicle: bool = False
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "event_id": self.event_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "location": self.location,
            "priority": self.priority,
            "requires_vehicle": self.requires_vehicle,
            "description": self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScheduleEvent":
        """从字典创建"""
        return cls(
            event_id=data["event_id"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]),
            location=data["location"],
            priority=data.get("priority", 5),
            requires_vehicle=data.get("requires_vehicle", False),
            description=data.get("description", "")
        )
    
    def conflicts_with(self, other: "ScheduleEvent") -> bool:
        """检查是否与另一个事件冲突
        
        Args:
            other: 另一个日程事件
            
        Returns:
            是否冲突
        """
        return (
            self.start_time < other.end_time and
            self.end_time > other.start_time
        )


@dataclass
class UserSchedule:
    """用户日程
    
    Attributes:
        user_id: 用户ID
        vehicle_id: 车辆ID
        events: 日程事件列表
        required_soc: 目标SOC (0-1)
        required_departure: 需用车时间
        flexibility: 时间灵活度 (小时)
        created_at: 创建时间
        updated_at: 更新时间
    """
    user_id: str
    vehicle_id: str
    events: List[ScheduleEvent] = field(default_factory=list)
    required_soc: float = 0.8
    required_departure: Optional[datetime] = None
    flexibility: float = 1.0  # 默认1小时灵活度
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "user_id": self.user_id,
            "vehicle_id": self.vehicle_id,
            "events": [e.to_dict() for e in self.events],
            "required_soc": self.required_soc,
            "required_departure": self.required_departure.isoformat() if self.required_departure else None,
            "flexibility": self.flexibility,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserSchedule":
        """从字典创建"""
        return cls(
            user_id=data["user_id"],
            vehicle_id=data["vehicle_id"],
            events=[ScheduleEvent.from_dict(e) for e in data.get("events", [])],
            required_soc=data.get("required_soc", 0.8),
            required_departure=datetime.fromisoformat(data["required_departure"]) if data.get("required_departure") else None,
            flexibility=data.get("flexibility", 1.0),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat()))
        )
    
    def get_next_vehicle_need(self) -> Optional[ScheduleEvent]:
        """获取下一个需要用车的事件
        
        Returns:
            下一个需要用车的事件，如果没有则返回None
        """
        now = datetime.now()
        future_events = [
            e for e in self.events
            if e.start_time > now and e.requires_vehicle
        ]
        if not future_events:
            return None
        return min(future_events, key=lambda e: e.start_time)
    
    def get_charging_windows(
        self,
        min_duration: timedelta = timedelta(minutes=30)
    ) -> List[tuple]:
        """获取可用的充电时间窗口
        
        Args:
            min_duration: 最小窗口时长
            
        Returns:
            (开始时间, 结束时间) 列表
        """
        now = datetime.now()
        
        # 按开始时间排序事件
        sorted_events = sorted(self.events, key=lambda e: e.start_time)
        
        windows = []
        current_time = now
        
        for event in sorted_events:
            if event.requires_vehicle:
                # 需要用车的事件前需要预留充电时间
                if event.start_time - current_time >= min_duration:
                    windows.append((current_time, event.start_time))
                current_time = event.end_time
        
        # 最后一个事件之后的时间窗口
        if self.required_departure and current_time < self.required_departure:
            if self.required_departure - current_time >= min_duration:
                windows.append((current_time, self.required_departure))
        
        return windows
    
    def is_urgent(self, current_soc: float, charge_power: float = 7.0) -> bool:
        """判断是否为紧急充电需求
        
        Args:
            current_soc: 当前SOC
            charge_power: 充电功率 (kW)
            
        Returns:
            是否紧急
        """
        if not self.required_departure:
            return False
        
        if current_soc >= self.required_soc:
            return False
        
        # 计算所需充电时间
        energy_needed = (self.required_soc - current_soc) * 60  # 假设电池容量60kWh
        charge_time_needed = energy_needed / charge_power  # 小时
        
        # 可用时间
        time_available = (self.required_departure - datetime.now()).total_seconds() / 3600
        
        # 需用车时间 - 当前时间 < 充电所需时间 + 缓冲(1小时)
        return time_available < charge_time_needed + 1.0


class UserScheduleManager:
    """用户日程管理器
    
    管理多个用户的日程，处理日程变更通知。
    
    Attributes:
        schedules: 用户日程字典
        change_listeners: 变更监听器列表
        
    Example:
        >>> manager = UserScheduleManager()
        >>> manager.add_schedule("user_001", schedule)
        >>> requirements = manager.get_charging_requirements("user_001")
    """
    
    def __init__(self) -> None:
        """初始化管理器"""
        self._schedules: Dict[str, UserSchedule] = {}
        self._change_listeners: List[Callable[[str, UserSchedule, str], None]] = []
        self._lock = threading.RLock()
        
        logger.info("初始化用户日程管理器")
    
    def add_schedule(self, user_id: str, schedule: UserSchedule) -> bool:
        """添加用户日程
        
        Args:
            user_id: 用户ID
            schedule: 用户日程
            
        Returns:
            是否成功
        """
        with self._lock:
            is_update = user_id in self._schedules
            self._schedules[user_id] = schedule
            
            action = "updated" if is_update else "added"
            logger.info(f"用户日程 {action}: {user_id}")
            
            # 通知监听器
            self._notify_change(user_id, schedule, action)
            
            return True
    
    def get_schedule(self, user_id: str) -> Optional[UserSchedule]:
        """获取用户日程
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户日程或None
        """
        with self._lock:
            return self._schedules.get(user_id)
    
    def remove_schedule(self, user_id: str) -> bool:
        """移除用户日程
        
        Args:
            user_id: 用户ID
            
        Returns:
            是否成功
        """
        with self._lock:
            if user_id in self._schedules:
                schedule = self._schedules.pop(user_id)
                logger.info(f"用户日程移除: {user_id}")
                self._notify_change(user_id, schedule, "removed")
                return True
            return False
    
    def update_schedule(
        self,
        user_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """更新用户日程
        
        Args:
            user_id: 用户ID
            updates: 更新字段
            
        Returns:
            是否成功
        """
        with self._lock:
            schedule = self._schedules.get(user_id)
            if not schedule:
                logger.warning(f"更新失败，用户日程不存在: {user_id}")
                return False
            
            # 更新字段
            if "required_soc" in updates:
                schedule.required_soc = updates["required_soc"]
            if "required_departure" in updates:
                schedule.required_departure = updates["required_departure"]
            if "flexibility" in updates:
                schedule.flexibility = updates["flexibility"]
            if "events" in updates:
                schedule.events = updates["events"]
            
            schedule.updated_at = datetime.now()
            
            logger.info(f"用户日程更新: {user_id}")
            self._notify_change(user_id, schedule, "updated")
            
            return True
    
    def add_event(self, user_id: str, event: ScheduleEvent) -> bool:
        """添加日程事件
        
        Args:
            user_id: 用户ID
            event: 日程事件
            
        Returns:
            是否成功
        """
        with self._lock:
            schedule = self._schedules.get(user_id)
            if not schedule:
                logger.warning(f"添加事件失败，用户日程不存在: {user_id}")
                return False
            
            schedule.events.append(event)
            schedule.updated_at = datetime.now()
            
            logger.info(f"日程事件添加: {user_id}, {event.event_id}")
            self._notify_change(user_id, schedule, "event_added")
            
            return True
    
    def remove_event(self, user_id: str, event_id: str) -> bool:
        """移除日程事件
        
        Args:
            user_id: 用户ID
            event_id: 事件ID
            
        Returns:
            是否成功
        """
        with self._lock:
            schedule = self._schedules.get(user_id)
            if not schedule:
                return False
            
            original_count = len(schedule.events)
            schedule.events = [e for e in schedule.events if e.event_id != event_id]
            
            if len(schedule.events) < original_count:
                schedule.updated_at = datetime.now()
                logger.info(f"日程事件移除: {user_id}, {event_id}")
                self._notify_change(user_id, schedule, "event_removed")
                return True
            return False
    
    def get_charging_requirements(self, user_id: str) -> Optional[Dict[str, Any]]:
        """获取充电需求
        
        Args:
            user_id: 用户ID
            
        Returns:
            充电需求字典或None
        """
        with self._lock:
            schedule = self._schedules.get(user_id)
            if not schedule:
                return None
            
            next_need = schedule.get_next_vehicle_need()
            charging_windows = schedule.get_charging_windows()
            
            return {
                "user_id": schedule.user_id,
                "vehicle_id": schedule.vehicle_id,
                "required_soc": schedule.required_soc,
                "deadline": schedule.required_departure.isoformat() if schedule.required_departure else None,
                "flexibility": schedule.flexibility,
                "next_vehicle_need": next_need.to_dict() if next_need else None,
                "charging_windows": [
                    (start.isoformat(), end.isoformat())
                    for start, end in charging_windows
                ],
                "event_count": len(schedule.events)
            }
    
    def get_all_requirements(self) -> Dict[str, Dict[str, Any]]:
        """获取所有用户的充电需求
        
        Returns:
            用户ID到充电需求的映射
        """
        with self._lock:
            return {
                user_id: self.get_charging_requirements(user_id)
                for user_id in self._schedules.keys()
            }
    
    def get_urgent_users(
        self,
        user_soc_map: Dict[str, float],
        charge_power: float = 7.0
    ) -> List[str]:
        """获取有紧急需求的用户列表
        
        Args:
            user_soc_map: 用户ID到当前SOC的映射
            charge_power: 充电功率
            
        Returns:
            紧急用户ID列表
        """
        urgent_users = []
        
        with self._lock:
            for user_id, current_soc in user_soc_map.items():
                schedule = self._schedules.get(user_id)
                if schedule and schedule.is_urgent(current_soc, charge_power):
                    urgent_users.append(user_id)
        
        return urgent_users
    
    def add_change_listener(
        self,
        callback: Callable[[str, UserSchedule, str], None]
    ) -> None:
        """添加变更监听器
        
        Args:
            callback: 回调函数(user_id, schedule, action)
        """
        self._change_listeners.append(callback)
    
    def remove_change_listener(
        self,
        callback: Callable[[str, UserSchedule, str], None]
    ) -> None:
        """移除变更监听器
        
        Args:
            callback: 回调函数
        """
        if callback in self._change_listeners:
            self._change_listeners.remove(callback)
    
    def _notify_change(
        self,
        user_id: str,
        schedule: UserSchedule,
        action: str
    ) -> None:
        """通知变更
        
        Args:
            user_id: 用户ID
            schedule: 用户日程
            action: 操作类型
        """
        for listener in self._change_listeners:
            try:
                listener(user_id, schedule, action)
            except Exception as e:
                logger.error(f"变更监听器执行失败: {e}")
    
    def predict_charging_demand(
        self,
        time_horizon: timedelta = timedelta(hours=24)
    ) -> Dict[str, Any]:
        """预测充电需求
        
        Args:
            time_horizon: 预测时间范围
            
        Returns:
            需求预测结果
        """
        now = datetime.now()
        end_time = now + time_horizon
        
        hourly_demand = {}
        
        with self._lock:
            for schedule in self._schedules.values():
                for event in schedule.events:
                    if event.requires_vehicle and event.start_time <= end_time:
                        # 计算需要充电的小时
                        hour = event.start_time.hour
                        hourly_demand[hour] = hourly_demand.get(hour, 0) + 1
        
        return {
            "prediction_time": now.isoformat(),
            "time_horizon_hours": time_horizon.total_seconds() / 3600,
            "total_users": len(self._schedules),
            "hourly_demand": hourly_demand,
            "peak_hours": sorted(
                hourly_demand.keys(),
                key=lambda h: hourly_demand[h],
                reverse=True
            )[:3]
        }
    
    def export_to_json(self, filepath: str) -> bool:
        """导出到JSON文件
        
        Args:
            filepath: 文件路径
            
        Returns:
            是否成功
        """
        try:
            with self._lock:
                data = {
                    user_id: schedule.to_dict()
                    for user_id, schedule in self._schedules.items()
                }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"日程数据已导出: {filepath}")
            return True
        except Exception as e:
            logger.error(f"导出失败: {e}")
            return False
    
    def import_from_json(self, filepath: str) -> bool:
        """从JSON文件导入
        
        Args:
            filepath: 文件路径
            
        Returns:
            是否成功
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            with self._lock:
                self._schedules = {
                    user_id: UserSchedule.from_dict(schedule_data)
                    for user_id, schedule_data in data.items()
                }
            
            logger.info(f"日程数据已导入: {filepath}, 共{len(self._schedules)}个用户")
            return True
        except Exception as e:
            logger.error(f"导入失败: {e}")
            return False


if __name__ == "__main__":
    # 简单测试
    manager = UserScheduleManager()
    
    # 创建用户日程
    schedule = UserSchedule(
        user_id="user_001",
        vehicle_id="vehicle_001",
        required_soc=0.8,
        required_departure=datetime.now() + timedelta(hours=3),
        flexibility=1.5
    )
    
    # 添加事件
    schedule.events.append(ScheduleEvent(
        event_id="event_001",
        start_time=datetime.now() + timedelta(hours=4),
        end_time=datetime.now() + timedelta(hours=5),
        location="Office",
        requires_vehicle=True,
        priority=8
    ))
    
    # 添加到管理器
    manager.add_schedule("user_001", schedule)
    
    # 获取充电需求
    requirements = manager.get_charging_requirements("user_001")
    print(f"充电需求: {requirements}")
    
    # 检查是否紧急
    is_urgent = schedule.is_urgent(current_soc=0.3)
    print(f"是否紧急: {is_urgent}")
