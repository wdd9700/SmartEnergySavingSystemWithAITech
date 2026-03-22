"""
GPU类型定义模块

定义GPU管理所需的数据结构和枚举类型
"""

from dataclasses import dataclass
from typing import Optional, List
from enum import Enum


class GPUVendor(Enum):
    """GPU厂商枚举"""
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    UNKNOWN = "unknown"


class GPUPowerState(Enum):
    """GPU电源状态"""
    IDLE = "idle"             # 闲置状态 (< 5%)
    LIGHT = "light"           # 轻度负载 (5-30%)
    MODERATE = "moderate"     # 中度负载 (30-70%)
    HEAVY = "heavy"           # 重度负载 (> 70%)


@dataclass
class GPUInfo:
    """GPU信息数据类
    
    包含GPU的各项监控指标和状态信息
    """
    gpu_id: int
    vendor: GPUVendor
    name: str
    
    # 利用率
    gpu_util: float          # GPU利用率 (%)
    memory_util: float       # 显存利用率 (%)
    
    # 显存
    memory_used_mb: int
    memory_total_mb: int
    
    # 功耗
    power_draw_w: float      # 当前功耗 (W)
    power_limit_w: float     # 功耗限制 (W)
    power_max_w: float       # 最大功耗 (W)
    
    # 温度
    temperature: float       # 温度 (°C)
    
    # 频率
    clock_mhz: int           # 当前频率 (MHz)
    max_clock_mhz: int       # 最大频率 (MHz)
    
    def __post_init__(self):
        """初始化后处理，确保数值类型正确"""
        self.gpu_util = float(self.gpu_util)
        self.memory_util = float(self.memory_util)
        self.power_draw_w = float(self.power_draw_w)
        self.power_limit_w = float(self.power_limit_w)
        self.power_max_w = float(self.power_max_w)
        self.temperature = float(self.temperature)


@dataclass
class GPUOptimizationConfig:
    """GPU优化配置
    
    用于配置自动功耗优化的参数
    """
    # 功耗限制比例 (相对于最大功耗)
    idle_power_percent: float = 50.0      # 闲置时功耗限制 (%)
    light_power_percent: float = 70.0     # 轻度负载时功耗限制 (%)
    moderate_power_percent: float = 85.0  # 中度负载时功耗限制 (%)
    heavy_power_percent: float = 100.0    # 重度负载时功耗限制 (%)
    
    # 状态检测阈值
    idle_threshold: float = 5.0           # 闲置阈值 (%)
    light_threshold: float = 30.0         # 轻度负载阈值 (%)
    moderate_threshold: float = 70.0      # 中度负载阈值 (%)
    
    # 自动优化间隔
    check_interval_seconds: int = 60      # 检测间隔 (秒)
    
    # 温度保护
    max_temperature: float = 85.0         # 最大允许温度 (°C)
    thermal_throttle_enabled: bool = True # 是否启用温度保护


class GPUError(Exception):
    """GPU管理基础异常"""
    pass


class GPUInitializationError(GPUError):
    """GPU初始化失败异常"""
    pass


class GPUNotFoundError(GPUError):
    """GPU未找到异常"""
    pass


class GPUPowerLimitError(GPUError):
    """功耗限制设置失败异常"""
    pass


class GPUUnsupportedError(GPUError):
    """GPU不支持的操作异常"""
    pass
