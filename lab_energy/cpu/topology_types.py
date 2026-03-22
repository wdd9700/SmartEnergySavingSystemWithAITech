"""
CPU拓扑类型定义模块

定义CPU拓扑识别所需的数据结构和枚举类型
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from enum import Enum


class CPUVendor(Enum):
    """CPU厂商枚举"""
    AMD = "AuthenticAMD"
    INTEL = "GenuineIntel"
    UNKNOWN = "Unknown"


class CoreType(Enum):
    """Intel大小核类型"""
    P_CORE = "performance"      # 性能核
    E_CORE = "efficient"        # 能效核
    SOC_CORE = "soc"            # SoC低功耗核
    UNKNOWN = "unknown"


class CCDType(Enum):
    """AMD CCD类型"""
    CCD0_CACHE = "ccd0_cache"   # 大缓存CCD
    CCD1_FREQ = "ccd1_freq"     # 高频率CCD
    UNKNOWN = "unknown"


@dataclass
class CoreInfo:
    """单个核心信息"""
    core_id: int
    logical_processor_id: int
    numa_node: int
    l3_cache_id: int
    core_type: Optional[CoreType] = None      # Intel
    ccd_type: Optional[CCDType] = None        # AMD
    base_freq_mhz: float = 0.0
    max_freq_mhz: float = 0.0

    def __post_init__(self):
        """初始化后处理，确保数值类型正确"""
        self.base_freq_mhz = float(self.base_freq_mhz)
        self.max_freq_mhz = float(self.max_freq_mhz)


@dataclass
class CPUTopology:
    """CPU拓扑信息"""
    vendor: CPUVendor
    model_name: str
    physical_cores: int
    logical_cores: int
    numa_nodes: int
    l3_caches: int
    cores: List[CoreInfo] = field(default_factory=list)
    
    # AMD特定
    ccd_count: Optional[int] = None
    has_3d_vcache: bool = False
    
    # Intel特定
    p_core_count: Optional[int] = None
    e_core_count: Optional[int] = None
    soc_core_count: Optional[int] = None

    def __post_init__(self):
        """初始化后处理"""
        if self.cores is None:
            self.cores = []


class TaskType(Enum):
    """任务类型枚举"""
    INTERACTIVE = "interactive"         # 交互式任务
    COMPUTE = "compute"                 # 计算密集型任务
    BACKGROUND = "background"           # 后台任务
    CACHE_SENSITIVE = "cache_sensitive" # 缓存敏感任务
