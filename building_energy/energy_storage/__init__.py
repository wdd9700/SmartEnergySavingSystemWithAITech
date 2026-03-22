"""
储能管理系统 (Energy Storage Management System)

提供电池储能系统的建模、电价策略集成、电网状态感知，以及多目标优化调度算法。

主要组件:
- BatteryModel: 电池物理模型
- PriceAPI: 电价API接口
- EnergyScheduler: 储能调度优化器
- StorageController: 储能控制器

使用示例:
    >>> from building_energy.energy_storage import StorageController
    >>> controller = StorageController.from_config("config.yaml")
    >>> controller.run()
"""

from .battery_model import BatteryParams, BatteryState, BatteryModel
from .price_api import ElectricityPrice, PriceSchedule, PriceAPI
from .scheduler import Schedule, EnergyScheduler
from .controller import StorageController, PriceEvent, GridEvent

__version__ = "1.0.0"
__all__ = [
    "BatteryParams",
    "BatteryState",
    "BatteryModel",
    "ElectricityPrice",
    "PriceSchedule",
    "PriceAPI",
    "Schedule",
    "EnergyScheduler",
    "StorageController",
    "PriceEvent",
    "GridEvent",
]
