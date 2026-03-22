"""
CPU拓扑识别与亲和度管理系统

自动识别AMD CCD架构和Intel大小核架构，为亲和度调度提供硬件信息。
支持Windows CPU Sets API和Thread Affinity API的进程/线程亲和度控制。

支持:
- AMD Ryzen 7000/9000系列 (CCD架构)
- Intel Core 12代+ (大小核架构)
- Intel Core Ultra (SoC Core)

示例:
    >>> from lab_energy.cpu import CPUTopologyDetector, CPUAffinityManager
    >>> detector = CPUTopologyDetector()
    >>> topology = detector.detect()
    >>> print(f"CPU: {topology.model_name}")
    >>> 
    >>> # 获取推荐亲和度
    >>> interactive_cores = detector.get_recommended_affinity("interactive")
    >>> background_cores = detector.get_recommended_affinity("background")
    >>>>
    >>> # 使用亲和度管理器
    >>> manager = CPUAffinityManager(detector)
    >>> manager.isolate_background_process(12345)  # 隔离后台进程
    >>> manager.optimize_for_interactive(67890)    # 优化交互式应用
"""

from .topology_types import (
    CPUVendor,
    CoreType,
    CCDType,
    CoreInfo,
    CPUTopology,
    TaskType,
)

from .topology import (
    CPUTopologyDetector,
    detect_topology,
    get_detector,
)

from .amd_detector import AMDCCDDetector
from .intel_detector import IntelHybridDetector

# 亲和度管理模块
from .affinity_types import (
    TaskType as AffinityTaskType,
    AffinityPolicy,
    CPUSetInfo,
    ThreadAffinityInfo,
    ProcessAffinityInfo,
)

from .affinity_api import (
    WindowsAffinityAPI,
    get_affinity_api,
)

from .affinity_manager import (
    CPUAffinityManager,
    get_affinity_manager,
)

# 激进Boost控制系统
from .boost_controller import (
    BoostController,
    PowerMode,
    BoostProfile,
    GameModeAPI,
    CPUMonitor,
    BOOST_PROFILES,
    quick_boost,
    set_performance_mode,
    set_powersave_mode,
)

from .power_api import (
    WindowsPowerAPI,
    set_process_high_performance,
    boost_system_performance,
)

__version__ = "1.1.0"

__all__ = [
    # 主要类 - CPU拓扑
    "CPUTopologyDetector",
    "AMDCCDDetector",
    "IntelHybridDetector",
    # 主要类 - 亲和度管理
    "CPUAffinityManager",
    "WindowsAffinityAPI",
    # 数据类型 - CPU拓扑
    "CPUVendor",
    "CoreType",
    "CCDType",
    "CoreInfo",
    "CPUTopology",
    "TaskType",
    # 数据类型 - 亲和度管理
    "AffinityPolicy",
    "CPUSetInfo",
    "ThreadAffinityInfo",
    "ProcessAffinityInfo",
    # 便捷函数 - CPU拓扑
    "detect_topology",
    "get_detector",
    # 便捷函数 - 亲和度管理
    "get_affinity_api",
    "get_affinity_manager",
    # 激进Boost控制系统
    "BoostController",
    "PowerMode",
    "BoostProfile",
    "GameModeAPI",
    "CPUMonitor",
    "WindowsPowerAPI",
    # 预定义配置
    "BOOST_PROFILES",
    # 便捷函数 - Boost控制
    "quick_boost",
    "set_performance_mode",
    "set_powersave_mode",
    "set_process_high_performance",
    "boost_system_performance",
]
