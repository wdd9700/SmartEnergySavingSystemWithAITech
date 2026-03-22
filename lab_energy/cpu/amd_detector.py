"""
AMD CCD架构检测模块

支持AMD Ryzen 7000/9000系列处理器的CCD拓扑识别
特别支持9950X3D等双CCD处理器（CCD0大缓存/CCD1高频率）
"""

import ctypes
from ctypes import wintypes
from typing import List, Dict, Optional, Set, Tuple
import logging

try:
    from .topology_types import CoreInfo, CCDType, CPUVendor
except ImportError:
    from topology_types import CoreInfo, CCDType, CPUVendor

# 设置日志
logger = logging.getLogger(__name__)


# Windows API常量
ERROR_INSUFFICIENT_BUFFER = 122
RelationProcessorCore = 0
RelationNumaNode = 1
RelationCache = 2
RelationProcessorPackage = 3
RelationGroup = 4
RelationAll = 0xFFFF

# 缓存类型
CacheUnified = 0
CacheInstruction = 1
CacheData = 2
CacheTrace = 3


class SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX(ctypes.Structure):
    """Windows系统逻辑处理器信息扩展结构"""
    _fields_ = [
        ("Relationship", wintypes.DWORD),
        ("Size", wintypes.DWORD),
        ("Processor", ctypes.c_byte * 1),  # 变长结构，实际需要动态处理
    ]


class PROCESSOR_RELATIONSHIP(ctypes.Structure):
    """处理器关系结构"""
    _fields_ = [
        ("Flags", wintypes.BYTE),
        ("EfficiencyClass", wintypes.BYTE),
        ("Reserved", wintypes.BYTE * 20),
        ("GroupCount", wintypes.WORD),
        ("GroupMask", wintypes.WORD * 1),
    ]


class NUMA_NODE_RELATIONSHIP(ctypes.Structure):
    """NUMA节点关系结构"""
    _fields_ = [
        ("NodeNumber", wintypes.DWORD),
        ("Reserved", wintypes.BYTE * 20),
        ("GroupMask", wintypes.WORD * 1),
    ]


class CACHE_RELATIONSHIP(ctypes.Structure):
    """缓存关系结构"""
    _fields_ = [
        ("Level", wintypes.BYTE),
        ("Associativity", wintypes.BYTE),
        ("LineSize", wintypes.WORD),
        ("CacheSize", wintypes.DWORD),
        ("Type", wintypes.DWORD),
        ("Reserved", wintypes.BYTE * 20),
        ("GroupMask", wintypes.WORD * 1),
    ]


# 定义ULONG_PTR类型（在ctypes.wintypes中不存在）
if ctypes.sizeof(ctypes.c_void_p) == 8:
    ULONG_PTR = ctypes.c_ulonglong
else:
    ULONG_PTR = ctypes.c_ulong


class GROUP_AFFINITY(ctypes.Structure):
    """组亲和性结构"""
    _fields_ = [
        ("Mask", ULONG_PTR),
        ("Group", wintypes.WORD),
        ("Reserved", wintypes.WORD * 3),
    ]


class PROCESSOR_GROUP_INFO(ctypes.Structure):
    """处理器组信息结构"""
    _fields_ = [
        ("MaximumProcessorCount", wintypes.BYTE),
        ("ActiveProcessorCount", wintypes.BYTE),
        ("Reserved", wintypes.BYTE * 38),
        ("ActiveProcessorMask", ULONG_PTR),
    ]


class AMDCCDDetector:
    """
    AMD CCD架构检测器
    
    支持识别AMD Ryzen 7000/9000系列处理器的CCD布局
    特别是双CCD处理器如9950X3D（CCD0大缓存/CCD1高频率）
    """
    
    # 已知支持3D V-Cache的AMD处理器型号
    X3D_MODELS = {
        "7800X3D", "7900X3D", "7950X3D", "9800X3D", "9900X3D", "9950X3D"
    }
    
    def __init__(self):
        self.kernel32 = ctypes.windll.kernel32
        self._topology_cache: Optional[Dict] = None
    
    def detect_ccd_topology(self, model_name: str, cores: List[CoreInfo]) -> Dict:
        """
        检测AMD CCD拓扑
        
        Args:
            model_name: CPU型号名称
            cores: 核心信息列表
            
        Returns:
            CCD拓扑信息字典
        """
        result = {
            "ccd_count": 0,
            "has_3d_vcache": False,
            "ccd0_cores": [],
            "ccd1_cores": [],
            "ccd_mapping": {}
        }
        
        # 检测是否为X3D处理器
        result["has_3d_vcache"] = self._is_x3d_processor(model_name)
        
        # 通过L3缓存布局推断CCD
        ccd_info = self._infer_ccd_from_cache(cores)
        result["ccd_count"] = ccd_info["count"]
        result["ccd_mapping"] = ccd_info["mapping"]
        result["ccd0_cores"] = ccd_info["ccd0_cores"]
        result["ccd1_cores"] = ccd_info["ccd1_cores"]
        
        # 更新cores的ccd_type
        self._update_core_ccd_types(cores, ccd_info["mapping"])
        
        return result
    
    def _is_x3d_processor(self, model_name: str) -> bool:
        """
        检测是否为X3D处理器
        
        Args:
            model_name: CPU型号名称
            
        Returns:
            是否为X3D处理器
        """
        model_upper = model_name.upper()
        for x3d_model in self.X3D_MODELS:
            if x3d_model in model_upper:
                return True
        return False
    
    def _infer_ccd_from_cache(self, cores: List[CoreInfo]) -> Dict:
        """
        通过L3缓存布局推断CCD结构
        
        AMD处理器通常每个CCD有独立的L3缓存
        
        Args:
            cores: 核心信息列表
            
        Returns:
            CCD推断结果
        """
        # 按L3缓存ID分组
        l3_groups: Dict[int, List[int]] = {}
        for core in cores:
            l3_id = core.l3_cache_id
            if l3_id not in l3_groups:
                l3_groups[l3_id] = []
            l3_groups[l3_id].append(core.core_id)
        
        ccd_count = len(l3_groups)
        mapping: Dict[int, CCDType] = {}
        ccd0_cores: List[int] = []
        ccd1_cores: List[int] = []
        
        # 按L3缓存ID排序，通常ID小的对应CCD0
        sorted_l3_ids = sorted(l3_groups.keys())
        
        for idx, l3_id in enumerate(sorted_l3_ids):
            core_ids = l3_groups[l3_id]
            
            if idx == 0:
                # CCD0 - 通常是主CCD，X3D处理器上带有3D V-Cache
                for core_id in core_ids:
                    mapping[core_id] = CCDType.CCD0_CACHE
                ccd0_cores = sorted(core_ids)
            elif idx == 1:
                # CCD1 - 通常是第二个CCD，高频率
                for core_id in core_ids:
                    mapping[core_id] = CCDType.CCD1_FREQ
                ccd1_cores = sorted(core_ids)
            else:
                # 更多CCD（如线程撕裂者）
                ccd_type = CCDType.CCD0_CACHE if idx % 2 == 0 else CCDType.CCD1_FREQ
                for core_id in core_ids:
                    mapping[core_id] = ccd_type
        
        return {
            "count": ccd_count,
            "mapping": mapping,
            "ccd0_cores": sorted(ccd0_cores),
            "ccd1_cores": sorted(ccd1_cores)
        }
    
    def _update_core_ccd_types(self, cores: List[CoreInfo], mapping: Dict[int, CCDType]):
        """
        更新核心CCD类型
        
        Args:
            cores: 核心信息列表
            mapping: 核心ID到CCD类型的映射
        """
        for core in cores:
            if core.core_id in mapping:
                core.ccd_type = mapping[core.core_id]
            else:
                core.ccd_type = CCDType.UNKNOWN
    
    def get_ccd0_cores(self, cores: List[CoreInfo]) -> List[int]:
        """
        获取CCD0（大缓存）的核心ID列表
        
        Args:
            cores: 核心信息列表
            
        Returns:
            CCD0核心ID列表
        """
        return sorted([c.core_id for c in cores if c.ccd_type == CCDType.CCD0_CACHE])
    
    def get_ccd1_cores(self, cores: List[CoreInfo]) -> List[int]:
        """
        获取CCD1（高频率）的核心ID列表
        
        Args:
            cores: 核心信息列表
            
        Returns:
            CCD1核心ID列表
        """
        return sorted([c.core_id for c in cores if c.ccd_type == CCDType.CCD1_FREQ])
    
    def get_recommended_affinity(self, cores: List[CoreInfo], 
                                  task_type: str) -> Set[int]:
        """
        根据任务类型推荐CPU亲和度
        
        Args:
            cores: 核心信息列表
            task_type: 任务类型
            
        Returns:
            推荐的核心ID集合
        """
        ccd0_cores = self.get_ccd0_cores(cores)
        ccd1_cores = self.get_ccd1_cores(cores)
        all_cores = set(ccd0_cores + ccd1_cores)
        
        if task_type == "interactive":
            # 交互式任务优先使用CCD1（高频率）
            if ccd1_cores:
                return set(ccd1_cores[:4])  # 使用前4个高频核心
            return set(list(all_cores)[:4])
            
        elif task_type == "compute":
            # 计算密集型任务使用所有核心
            return all_cores
            
        elif task_type == "background":
            # 后台任务优先使用CCD0（如果存在）
            if ccd0_cores:
                return set(ccd0_cores)
            return all_cores
            
        elif task_type == "cache_sensitive":
            # 缓存敏感任务使用CCD0（大缓存）
            if ccd0_cores:
                return set(ccd0_cores)
            return all_cores
            
        else:
            # 默认使用所有核心
            return all_cores
    
    def estimate_ccd_frequencies(self, model_name: str) -> Dict[str, float]:
        """
        估算CCD频率
        
        基于已知型号的典型频率值
        
        Args:
            model_name: CPU型号名称
            
        Returns:
            CCD频率估算值
        """
        model_upper = model_name.upper()
        
        # 9950X3D典型频率
        if "9950X3D" in model_upper:
            return {
                "ccd0_base": 4200.0,   # CCD0基础频率
                "ccd0_max": 5250.0,    # CCD0最大频率（带3D V-Cache）
                "ccd1_base": 4200.0,   # CCD1基础频率
                "ccd1_max": 5700.0     # CCD1最大频率（更高）
            }
        
        # 7950X3D典型频率
        elif "7950X3D" in model_upper:
            return {
                "ccd0_base": 4200.0,
                "ccd0_max": 5250.0,
                "ccd1_base": 4200.0,
                "ccd1_max": 5700.0
            }
        
        # 7900X3D典型频率
        elif "7900X3D" in model_upper:
            return {
                "ccd0_base": 4400.0,
                "ccd0_max": 5600.0,
                "ccd1_base": 4400.0,
                "ccd1_max": 5600.0
            }
        
        # 7800X3D典型频率（单CCD）
        elif "7800X3D" in model_upper:
            return {
                "ccd0_base": 4200.0,
                "ccd0_max": 5000.0,
                "ccd1_base": 0.0,
                "ccd1_max": 0.0
            }
        
        # 默认频率估算
        return {
            "ccd0_base": 4000.0,
            "ccd0_max": 5000.0,
            "ccd1_base": 4000.0,
            "ccd1_max": 5500.0
        }
