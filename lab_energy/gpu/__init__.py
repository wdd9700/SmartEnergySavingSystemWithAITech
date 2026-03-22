"""
GPU节能管理系统

提供NVIDIA和AMD显卡的监控与功耗控制功能

使用示例:
    >>> from lab_energy.gpu import GPUManager, GPUVendor, GPUPowerState
    >>> 
    >>> # 创建并初始化GPU管理器
    >>> gpu = GPUManager()
    >>> if gpu.initialize():
    ...     print(f"检测到 {gpu.get_gpu_count()} 个GPU")
    ...     
    ...     # 获取GPU信息
    ...     info = gpu.get_gpu_info(0)
    ...     print(f"GPU: {info.name}")
    ...     print(f"利用率: {info.gpu_util}%")
    ...     print(f"功耗: {info.power_draw_w:.1f}W")
    ...     
    ...     # 手动设置功耗限制
    ...     gpu.set_power_limit(0, 150.0)  # 限制到150W
    ...     
    ...     # 自动优化
    ...     gpu.auto_optimize(interval_seconds=60)
    ...     
    ...     # 关闭管理器
    ...     gpu.shutdown()
"""

from .gpu_types import (
    GPUVendor,
    GPUPowerState,
    GPUInfo,
    GPUOptimizationConfig,
    GPUError,
    GPUInitializationError,
    GPUNotFoundError,
    GPUPowerLimitError,
    GPUUnsupportedError,
)

from .gpu_manager import GPUManager, GPUStats

__all__ = [
    # 类型定义
    'GPUVendor',
    'GPUPowerState',
    'GPUInfo',
    'GPUOptimizationConfig',
    'GPUStats',
    # 异常
    'GPUError',
    'GPUInitializationError',
    'GPUNotFoundError',
    'GPUPowerLimitError',
    'GPUUnsupportedError',
    # 主类
    'GPUManager',
]

__version__ = '1.0.0'
