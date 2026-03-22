"""
Intel大小核架构检测模块

支持Intel Core 12代+处理器的大小核拓扑识别
支持Intel Core Ultra的P-Core/E-Core/SoC Core识别
"""

import ctypes
from ctypes import wintypes
from typing import List, Dict, Optional, Set
import logging
import platform

try:
    from .topology_types import CoreInfo, CoreType, CPUVendor
except ImportError:
    from topology_types import CoreInfo, CoreType, CPUVendor

# 设置日志
logger = logging.getLogger(__name__)


# Windows API常量
ERROR_INSUFFICIENT_BUFFER = 122

# CPU Set API常量
SYSTEM_CPU_SET_INFORMATION_PARKED = 0x1
SYSTEM_CPU_SET_INFORMATION_ALLOCATED = 0x2
SYSTEM_CPU_SET_INFORMATION_ALLOCATED_TO_TARGET_PROCESS = 0x4
SYSTEM_CPU_SET_INFORMATION_REALTIME = 0x8


class CPU_SET_INFORMATION(ctypes.Structure):
    """CPU Set信息结构"""
    _fields_ = [
        ("Id", wintypes.DWORD),
        ("Group", wintypes.WORD),
        ("LogicalProcessorIndex", wintypes.BYTE),
        ("CoreIndex", wintypes.BYTE),
        ("LastLevelCacheIndex", wintypes.BYTE),
        ("NumaNodeIndex", wintypes.BYTE),
        ("EfficiencyClass", wintypes.BYTE),
        ("AllFlags", wintypes.BYTE),
        ("Type", wintypes.DWORD),
        ("Reserved", wintypes.DWORD),
    ]


class IntelHybridDetector:
    """
    Intel大小核架构检测器
    
    支持识别Intel Core 12代+处理器的大小核布局
    支持Intel Core Ultra的P-Core/E-Core/SoC Core识别
    使用Windows CPU Sets API获取效率等级信息
    """
    
    def __init__(self):
        self.kernel32 = ctypes.windll.kernel32
        self._cpu_sets_cache: Optional[List[Dict]] = None
        self._windows_version = self._get_windows_version()
    
    def _get_windows_version(self) -> tuple:
        """获取Windows版本信息"""
        try:
            version = platform.version()
            parts = version.split('.')
            if len(parts) >= 2:
                return (int(parts[0]), int(parts[1]))
        except Exception as e:
            logger.warning(f"无法获取Windows版本: {e}")
        return (10, 0)  # 默认值
    
    def _is_cpu_sets_supported(self) -> bool:
        """
        检查是否支持CPU Sets API
        
        CPU Sets API需要Windows 10 1703+ (版本号15063+)
        """
        major, build = self._windows_version
        # Windows 10 1703 build 15063
        return major >= 10 and build >= 15063
    
    def _get_cpu_sets(self) -> Optional[List[Dict]]:
        """
        使用CPU Sets API获取处理器信息
        
        Returns:
            CPU Set信息列表，如果API不支持则返回None
        """
        if not self._is_cpu_sets_supported():
            logger.debug("CPU Sets API不支持当前Windows版本")
            return None
        
        if self._cpu_sets_cache is not None:
            return self._cpu_sets_cache
        
        try:
            # 首先获取所需缓冲区大小
            buffer_size = wintypes.DWORD(0)
            result = self.kernel32.GetSystemCpuSetInformation(
                None, 0, ctypes.byref(buffer_size), None, 0
            )
            
            if result == 0 and ctypes.get_last_error() != ERROR_INSUFFICIENT_BUFFER:
                logger.warning(f"GetSystemCpuSetInformation失败: {ctypes.get_last_error()}")
                return None
            
            # 分配缓冲区
            buffer = ctypes.create_string_buffer(buffer_size.value)
            
            # 获取CPU Set信息
            result = self.kernel32.GetSystemCpuSetInformation(
                buffer, buffer_size.value, ctypes.byref(buffer_size), None, 0
            )
            
            if result == 0:
                logger.warning(f"GetSystemCpuSetInformation失败: {ctypes.get_last_error()}")
                return None
            
            # 解析CPU Set信息
            cpu_sets = []
            offset = 0
            
            while offset < buffer_size.value:
                # 读取CPU Set信息头部
                info_ptr = ctypes.cast(
                    ctypes.addressof(buffer) + offset, 
                    ctypes.POINTER(CPU_SET_INFORMATION)
                )
                info = info_ptr.contents
                
                cpu_set = {
                    "id": info.Id,
                    "group": info.Group,
                    "logical_processor_index": info.LogicalProcessorIndex,
                    "core_index": info.CoreIndex,
                    "llc_index": info.LastLevelCacheIndex,
                    "numa_node": info.NumaNodeIndex,
                    "efficiency_class": info.EfficiencyClass,
                    "flags": info.AllFlags,
                    "type": info.Type
                }
                cpu_sets.append(cpu_set)
                
                # 移动到下一个条目 (每个条目大小为CPU_SET_INFORMATION结构大小)
                offset += ctypes.sizeof(CPU_SET_INFORMATION)
            
            self._cpu_sets_cache = cpu_sets
            return cpu_sets
            
        except Exception as e:
            logger.warning(f"获取CPU Sets信息时出错: {e}")
            return None
    
    def detect_hybrid_topology(self, model_name: str, 
                               cores: List[CoreInfo]) -> Dict:
        """
        检测Intel大小核拓扑
        
        Args:
            model_name: CPU型号名称
            cores: 核心信息列表
            
        Returns:
            大小核拓扑信息字典
        """
        result = {
            "p_core_count": 0,
            "e_core_count": 0,
            "soc_core_count": 0,
            "p_cores": [],
            "e_cores": [],
            "soc_cores": [],
            "core_type_mapping": {}
        }
        
        # 首先尝试使用CPU Sets API
        cpu_sets = self._get_cpu_sets()
        
        if cpu_sets:
            # 使用CPU Sets API获取效率等级
            type_info = self._detect_from_cpu_sets(cpu_sets, cores)
        else:
            # 回退到基于型号名称的检测
            type_info = self._detect_from_model_name(model_name, cores)
        
        # 更新结果
        result["p_cores"] = type_info["p_cores"]
        result["e_cores"] = type_info["e_cores"]
        result["soc_cores"] = type_info["soc_cores"]
        result["p_core_count"] = len(type_info["p_cores"])
        result["e_core_count"] = len(type_info["e_cores"])
        result["soc_core_count"] = len(type_info["soc_cores"])
        result["core_type_mapping"] = type_info["mapping"]
        
        # 更新cores的core_type
        self._update_core_types(cores, type_info["mapping"])
        
        return result
    
    def _detect_from_cpu_sets(self, cpu_sets: List[Dict], 
                              cores: List[CoreInfo]) -> Dict:
        """
        从CPU Sets信息检测核心类型
        
        EfficiencyClass值含义:
        - 0: E-Core (能效核)
        - 1: P-Core (性能核)
        - 2+: SoC Core (SoC低功耗核，Intel Core Ultra)
        
        Args:
            cpu_sets: CPU Set信息列表
            cores: 核心信息列表
            
        Returns:
            核心类型信息
        """
        p_cores = []
        e_cores = []
        soc_cores = []
        mapping: Dict[int, CoreType] = {}
        
        # 创建逻辑处理器ID到核心ID的映射
        lp_to_core: Dict[int, int] = {}
        for core in cores:
            lp_to_core[core.logical_processor_id] = core.core_id
        
        for cpu_set in cpu_sets:
            lp_id = cpu_set["id"]
            efficiency = cpu_set["efficiency_class"]
            
            if lp_id in lp_to_core:
                core_id = lp_to_core[lp_id]
                
                if efficiency >= 2:
                    core_type = CoreType.SOC_CORE
                    soc_cores.append(core_id)
                elif efficiency == 1:
                    core_type = CoreType.P_CORE
                    p_cores.append(core_id)
                else:
                    core_type = CoreType.E_CORE
                    e_cores.append(core_id)
                
                mapping[core_id] = core_type
        
        return {
            "p_cores": sorted(set(p_cores)),
            "e_cores": sorted(set(e_cores)),
            "soc_cores": sorted(set(soc_cores)),
            "mapping": mapping
        }
    
    def _detect_from_model_name(self, model_name: str, 
                                 cores: List[CoreInfo]) -> Dict:
        """
        基于型号名称检测核心类型（回退方案）
        
        Args:
            model_name: CPU型号名称
            cores: 核心信息列表
            
        Returns:
            核心类型信息
        """
        model_upper = model_name.upper()
        
        # 核心数量配置（物理核心）
        config = self._parse_intel_config(model_upper)
        
        total_cores = len(cores)
        p_cores = []
        e_cores = []
        soc_cores = []
        mapping: Dict[int, CoreType] = {}
        
        if config:
            p_count = config.get("p_cores", 0)
            e_count = config.get("e_cores", 0)
            soc_count = config.get("soc_cores", 0)
            
            # 假设P-Core在前，E-Core在后
            sorted_cores = sorted(cores, key=lambda c: c.core_id)
            
            for i, core in enumerate(sorted_cores):
                if i < p_count:
                    core_type = CoreType.P_CORE
                    p_cores.append(core.core_id)
                elif i < p_count + e_count:
                    core_type = CoreType.E_CORE
                    e_cores.append(core.core_id)
                elif i < p_count + e_count + soc_count:
                    core_type = CoreType.SOC_CORE
                    soc_cores.append(core.core_id)
                else:
                    core_type = CoreType.UNKNOWN
                
                mapping[core.core_id] = core_type
        else:
            # 无法识别配置，所有核心标记为未知
            for core in cores:
                mapping[core.core_id] = CoreType.UNKNOWN
        
        return {
            "p_cores": sorted(p_cores),
            "e_cores": sorted(e_cores),
            "soc_cores": sorted(soc_cores),
            "mapping": mapping
        }
    
    def _parse_intel_config(self, model_name: str) -> Optional[Dict]:
        """
        解析Intel处理器配置
        
        Args:
            model_name: CPU型号名称（大写）
            
        Returns:
            核心配置字典
        """
        # Core Ultra系列 (Meteor Lake, Arrow Lake)
        ultra_configs = {
            "ULTRA 9 285K": {"p_cores": 8, "e_cores": 16, "soc_cores": 2},
            "ULTRA 9 275HX": {"p_cores": 8, "e_cores": 16, "soc_cores": 0},
            "ULTRA 7 265K": {"p_cores": 8, "e_cores": 12, "soc_cores": 2},
            "ULTRA 7 255HX": {"p_cores": 8, "e_cores": 12, "soc_cores": 0},
            "ULTRA 7 265": {"p_cores": 8, "e_cores": 12, "soc_cores": 2},
            "ULTRA 5 245K": {"p_cores": 6, "e_cores": 8, "soc_cores": 2},
            "ULTRA 5 235HX": {"p_cores": 6, "e_cores": 8, "soc_cores": 0},
            "ULTRA 5 225": {"p_cores": 6, "e_cores": 8, "soc_cores": 2},
        }
        
        # Core 14代 (Raptor Lake Refresh)
        gen14_configs = {
            "I9-14900K": {"p_cores": 8, "e_cores": 16, "soc_cores": 0},
            "I9-14900": {"p_cores": 8, "e_cores": 16, "soc_cores": 0},
            "I7-14700K": {"p_cores": 8, "e_cores": 12, "soc_cores": 0},
            "I7-14700": {"p_cores": 8, "e_cores": 12, "soc_cores": 0},
            "I5-14600K": {"p_cores": 6, "e_cores": 8, "soc_cores": 0},
            "I5-14600": {"p_cores": 6, "e_cores": 8, "soc_cores": 0},
            "I5-14400": {"p_cores": 6, "e_cores": 4, "soc_cores": 0},
        }
        
        # Core 13代 (Raptor Lake)
        gen13_configs = {
            "I9-13900K": {"p_cores": 8, "e_cores": 16, "soc_cores": 0},
            "I9-13900": {"p_cores": 8, "e_cores": 16, "soc_cores": 0},
            "I7-13700K": {"p_cores": 8, "e_cores": 8, "soc_cores": 0},
            "I7-13700": {"p_cores": 8, "e_cores": 8, "soc_cores": 0},
            "I5-13600K": {"p_cores": 6, "e_cores": 8, "soc_cores": 0},
            "I5-13600": {"p_cores": 6, "e_cores": 8, "soc_cores": 0},
            "I5-13500": {"p_cores": 6, "e_cores": 8, "soc_cores": 0},
            "I5-13400": {"p_cores": 6, "e_cores": 4, "soc_cores": 0},
        }
        
        # Core 12代 (Alder Lake)
        gen12_configs = {
            "I9-12900K": {"p_cores": 8, "e_cores": 8, "soc_cores": 0},
            "I9-12900": {"p_cores": 8, "e_cores": 8, "soc_cores": 0},
            "I7-12700K": {"p_cores": 8, "e_cores": 4, "soc_cores": 0},
            "I7-12700": {"p_cores": 8, "e_cores": 4, "soc_cores": 0},
            "I5-12600K": {"p_cores": 6, "e_cores": 4, "soc_cores": 0},
            "I5-12600": {"p_cores": 6, "e_cores": 4, "soc_cores": 0},
            "I5-12500": {"p_cores": 6, "e_cores": 0, "soc_cores": 0},
            "I5-12400": {"p_cores": 6, "e_cores": 0, "soc_cores": 0},
            "I3-12100": {"p_cores": 4, "e_cores": 0, "soc_cores": 0},
        }
        
        # 合并所有配置
        all_configs = {}
        all_configs.update(ultra_configs)
        all_configs.update(gen14_configs)
        all_configs.update(gen13_configs)
        all_configs.update(gen12_configs)
        
        # 查找匹配的配置
        for config_name, config in all_configs.items():
            if config_name in model_name:
                return config
        
        return None
    
    def _update_core_types(self, cores: List[CoreInfo], 
                           mapping: Dict[int, CoreType]):
        """
        更新核心类型
        
        Args:
            cores: 核心信息列表
            mapping: 核心ID到核心类型的映射
        """
        for core in cores:
            if core.core_id in mapping:
                core.core_type = mapping[core.core_id]
            else:
                core.core_type = CoreType.UNKNOWN
    
    def get_p_cores(self, cores: List[CoreInfo]) -> List[int]:
        """
        获取P-Core列表
        
        Args:
            cores: 核心信息列表
            
        Returns:
            P-Core核心ID列表
        """
        return sorted([c.core_id for c in cores if c.core_type == CoreType.P_CORE])
    
    def get_e_cores(self, cores: List[CoreInfo]) -> List[int]:
        """
        获取E-Core列表
        
        Args:
            cores: 核心信息列表
            
        Returns:
            E-Core核心ID列表
        """
        return sorted([c.core_id for c in cores if c.core_type == CoreType.E_CORE])
    
    def get_soc_cores(self, cores: List[CoreInfo]) -> List[int]:
        """
        获取SoC Core列表
        
        Args:
            cores: 核心信息列表
            
        Returns:
            SoC Core核心ID列表
        """
        return sorted([c.core_id for c in cores if c.core_type == CoreType.SOC_CORE])
    
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
        p_cores = self.get_p_cores(cores)
        e_cores = self.get_e_cores(cores)
        soc_cores = self.get_soc_cores(cores)
        
        all_cores = set(p_cores + e_cores + soc_cores)
        
        if task_type == "interactive":
            # 交互式任务优先使用P-Core
            if p_cores:
                return set(p_cores[:4])  # 使用前4个P-Core
            return set(list(all_cores)[:4])
            
        elif task_type == "compute":
            # 计算密集型任务使用P-Core
            if p_cores:
                return set(p_cores)
            return all_cores
            
        elif task_type == "background":
            # 后台任务优先使用E-Core或SoC Core
            if e_cores:
                return set(e_cores)
            elif soc_cores:
                return set(soc_cores)
            return all_cores
            
        elif task_type == "cache_sensitive":
            # 缓存敏感任务使用P-Core（通常有更大缓存）
            if p_cores:
                return set(p_cores)
            return all_cores
            
        else:
            # 默认使用所有核心
            return all_cores
