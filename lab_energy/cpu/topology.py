"""
CPU拓扑检测主模块

提供统一的CPU拓扑检测接口，支持AMD CCD架构和Intel大小核架构
"""

import ctypes
from ctypes import wintypes
from typing import List, Dict, Optional, Set
import platform
import logging
import time

try:
    from .topology_types import (
        CPUTopology, CoreInfo, CPUVendor, CoreType, CCDType, TaskType
    )
    from .amd_detector import AMDCCDDetector
    from .intel_detector import IntelHybridDetector
except ImportError:
    from topology_types import (
        CPUTopology, CoreInfo, CPUVendor, CoreType, CCDType, TaskType
    )
    from amd_detector import AMDCCDDetector
    from intel_detector import IntelHybridDetector

# 设置日志
logger = logging.getLogger(__name__)


# Windows API常量
ERROR_INSUFFICIENT_BUFFER = 122
RelationProcessorCore = 0
RelationNumaNode = 1
RelationCache = 2
RelationProcessorPackage = 3
RelationGroup = 4

# 定义ULONG_PTR类型（在ctypes.wintypes中不存在）
if ctypes.sizeof(ctypes.c_void_p) == 8:
    ULONG_PTR = ctypes.c_ulonglong
else:
    ULONG_PTR = ctypes.c_ulong


class SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX(ctypes.Structure):
    """Windows系统逻辑处理器信息扩展结构"""
    _fields_ = [
        ("Relationship", wintypes.DWORD),
        ("Size", wintypes.DWORD),
    ]


class PROCESSOR_RELATIONSHIP(ctypes.Structure):
    """处理器关系结构"""
    _fields_ = [
        ("Flags", wintypes.BYTE),
        ("EfficiencyClass", wintypes.BYTE),
        ("Reserved", wintypes.BYTE * 20),
        ("GroupCount", wintypes.WORD),
        ("GroupMask", ULONG_PTR),
    ]


class NUMA_NODE_RELATIONSHIP(ctypes.Structure):
    """NUMA节点关系结构"""
    _fields_ = [
        ("NodeNumber", wintypes.DWORD),
        ("Reserved", wintypes.BYTE * 20),
        ("GroupMask", ULONG_PTR),
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
        ("GroupMask", ULONG_PTR),
    ]


class GROUP_RELATIONSHIP(ctypes.Structure):
    """处理器组关系结构"""
    _fields_ = [
        ("MaximumGroupCount", wintypes.WORD),
        ("ActiveGroupCount", wintypes.WORD),
        ("Reserved", wintypes.BYTE * 20),
        ("GroupMask", ULONG_PTR),
    ]


class CPUTopologyDetector:
    """
    CPU拓扑检测器
    
    支持:
    - AMD Ryzen 7000/9000系列 (CCD架构)
    - Intel Core 12代+ (大小核架构)
    - Intel Core Ultra (SoC Core)
    
    使用Windows API获取处理器拓扑信息，包括NUMA节点、缓存布局等
    """
    
    def __init__(self):
        self.kernel32 = ctypes.windll.kernel32
        self._topology_cache: Optional[CPUTopology] = None
        self._cache_timestamp: float = 0
        self._cache_ttl: float = 60.0  # 缓存60秒
        
        # 子检测器
        self._amd_detector = AMDCCDDetector()
        self._intel_detector = IntelHybridDetector()
    
    def detect(self, use_cache: bool = True) -> CPUTopology:
        """
        检测CPU拓扑
        
        Args:
            use_cache: 是否使用缓存结果
            
        Returns:
            完整的CPU拓扑信息
        """
        # 检查缓存
        if use_cache and self._topology_cache is not None:
            if time.time() - self._cache_timestamp < self._cache_ttl:
                return self._topology_cache
        
        start_time = time.time()
        
        # 获取基础处理器信息
        vendor = self._get_cpu_vendor()
        model_name = self._get_cpu_model_name()
        
        # 获取处理器拓扑信息
        proc_info = self._get_processor_topology()
        
        # 创建核心信息列表
        cores = self._create_core_info(proc_info)
        
        # 创建拓扑对象
        topology = CPUTopology(
            vendor=vendor,
            model_name=model_name,
            physical_cores=proc_info.get("physical_cores", 0),
            logical_cores=proc_info.get("logical_cores", 0),
            numa_nodes=proc_info.get("numa_nodes", 0),
            l3_caches=proc_info.get("l3_caches", 0),
            cores=cores
        )
        
        # 根据厂商进行特定检测
        if vendor == CPUVendor.AMD:
            self._detect_amd_topology(topology)
        elif vendor == CPUVendor.INTEL:
            self._detect_intel_topology(topology)
        
        # 更新缓存
        self._topology_cache = topology
        self._cache_timestamp = time.time()
        
        elapsed = time.time() - start_time
        logger.debug(f"CPU拓扑检测完成，耗时: {elapsed:.3f}s")
        
        return topology
    
    def _get_cpu_vendor(self) -> CPUVendor:
        """
        获取CPU厂商
        
        Returns:
            CPU厂商枚举
        """
        try:
            # 使用platform模块获取处理器信息
            processor = platform.processor()
            
            if "AuthenticAMD" in processor or "amd" in processor.lower():
                return CPUVendor.AMD
            elif "GenuineIntel" in processor or "intel" in processor.lower():
                return CPUVendor.INTEL
            
            # 回退到环境变量检测
            import os
            if "AMD" in os.environ.get("PROCESSOR_IDENTIFIER", ""):
                return CPUVendor.AMD
            elif "Intel" in os.environ.get("PROCESSOR_IDENTIFIER", ""):
                return CPUVendor.INTEL
                
        except Exception as e:
            logger.warning(f"获取CPU厂商失败: {e}")
        
        return CPUVendor.UNKNOWN
    
    def _get_cpu_model_name(self) -> str:
        """
        获取CPU型号名称
        
        Returns:
            CPU型号名称
        """
        try:
            # 使用platform模块
            processor = platform.processor()
            if processor:
                return processor
            
            # 回退到环境变量
            import os
            identifier = os.environ.get("PROCESSOR_IDENTIFIER", "")
            if identifier:
                return identifier
                
        except Exception as e:
            logger.warning(f"获取CPU型号失败: {e}")
        
        return "Unknown"
    
    def _get_processor_topology(self) -> Dict:
        """
        使用Windows API获取处理器拓扑
        
        Returns:
            处理器拓扑信息字典
        """
        result = {
            "physical_cores": 0,
            "logical_cores": 0,
            "numa_nodes": 0,
            "l3_caches": 0,
            "cores": {},
            "numa_nodes_info": {},
            "l3_caches_info": {}
        }
        
        try:
            # 获取逻辑处理器数量
            result["logical_cores"] = self.kernel32.GetActiveProcessorCount(0)
            
            # 使用GetLogicalProcessorInformationEx获取详细拓扑
            buffer_size = wintypes.DWORD(0)
            
            # 第一次调用获取缓冲区大小
            self.kernel32.GetLogicalProcessorInformationEx(
                RelationAll, None, ctypes.byref(buffer_size)
            )
            
            if buffer_size.value == 0:
                logger.warning("无法获取处理器拓扑信息")
                return result
            
            # 分配缓冲区
            buffer = ctypes.create_string_buffer(buffer_size.value)
            
            # 第二次调用获取数据
            success = self.kernel32.GetLogicalProcessorInformationEx(
                RelationAll, buffer, ctypes.byref(buffer_size)
            )
            
            if not success:
                logger.warning(f"GetLogicalProcessorInformationEx失败: {ctypes.get_last_error()}")
                return result
            
            # 解析数据
            offset = 0
            core_ids = set()
            
            while offset < buffer_size.value:
                info_ptr = ctypes.cast(
                    ctypes.addressof(buffer) + offset,
                    ctypes.POINTER(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)
                )
                info = info_ptr.contents
                
                if info.Relationship == RelationProcessorCore:
                    # 处理器核心
                    core_info = ctypes.cast(
                        ctypes.addressof(buffer) + offset + ctypes.sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX),
                        ctypes.POINTER(PROCESSOR_RELATIONSHIP)
                    ).contents
                    
                    core_id = len(core_ids)
                    core_ids.add(core_id)
                    
                    # 解析组掩码获取逻辑处理器ID
                    mask = core_info.GroupMask
                    lp_ids = self._parse_mask(mask)
                    
                    result["cores"][core_id] = {
                        "logical_processors": lp_ids,
                        "flags": core_info.Flags,
                        "efficiency_class": core_info.EfficiencyClass
                    }
                    
                elif info.Relationship == RelationNumaNode:
                    # NUMA节点
                    numa_info = ctypes.cast(
                        ctypes.addressof(buffer) + offset + ctypes.sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX),
                        ctypes.POINTER(NUMA_NODE_RELATIONSHIP)
                    ).contents
                    
                    node_id = numa_info.NodeNumber
                    mask = numa_info.GroupMask
                    lp_ids = self._parse_mask(mask)
                    
                    result["numa_nodes_info"][node_id] = lp_ids
                    
                elif info.Relationship == RelationCache:
                    # 缓存
                    cache_info = ctypes.cast(
                        ctypes.addressof(buffer) + offset + ctypes.sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX),
                        ctypes.POINTER(CACHE_RELATIONSHIP)
                    ).contents
                    
                    if cache_info.Level == 3:
                        # L3缓存
                        l3_id = result["l3_caches"]
                        mask = cache_info.GroupMask
                        lp_ids = self._parse_mask(mask)
                        
                        result["l3_caches_info"][l3_id] = {
                            "logical_processors": lp_ids,
                            "size": cache_info.CacheSize,
                            "associativity": cache_info.Associativity,
                            "line_size": cache_info.LineSize
                        }
                        result["l3_caches"] += 1
                
                # 移动到下一个条目
                offset += info.Size
            
            result["physical_cores"] = len(core_ids)
            result["numa_nodes"] = len(result["numa_nodes_info"])
            
        except Exception as e:
            logger.warning(f"获取处理器拓扑时出错: {e}")
        
        return result
    
    def _parse_mask(self, mask: int) -> List[int]:
        """
        解析处理器掩码为逻辑处理器ID列表
        
        Args:
            mask: 处理器掩码
            
        Returns:
            逻辑处理器ID列表
        """
        lp_ids = []
        for i in range(64):  # 假设最多64个逻辑处理器
            if mask & (1 << i):
                lp_ids.append(i)
        return lp_ids
    
    def _create_core_info(self, proc_info: Dict) -> List[CoreInfo]:
        """
        根据处理器信息创建核心信息列表
        
        Args:
            proc_info: 处理器拓扑信息
            
        Returns:
            核心信息列表
        """
        cores = []
        
        for core_id, core_data in proc_info.get("cores", {}).items():
            # 获取该核心的第一个逻辑处理器ID
            lp_ids = core_data.get("logical_processors", [0])
            lp_id = lp_ids[0] if lp_ids else 0
            
            # 确定NUMA节点
            numa_node = 0
            for node_id, node_lp_ids in proc_info.get("numa_nodes_info", {}).items():
                if lp_id in node_lp_ids:
                    numa_node = node_id
                    break
            
            # 确定L3缓存ID
            l3_cache_id = 0
            for l3_id, l3_data in proc_info.get("l3_caches_info", {}).items():
                if lp_id in l3_data.get("logical_processors", []):
                    l3_cache_id = l3_id
                    break
            
            core_info = CoreInfo(
                core_id=core_id,
                logical_processor_id=lp_id,
                numa_node=numa_node,
                l3_cache_id=l3_cache_id,
                base_freq_mhz=0.0,
                max_freq_mhz=0.0
            )
            cores.append(core_info)
        
        # 如果没有获取到核心信息，创建默认信息
        if not cores:
            logical_cores = proc_info.get("logical_cores", 1)
            for i in range(logical_cores):
                cores.append(CoreInfo(
                    core_id=i,
                    logical_processor_id=i,
                    numa_node=0,
                    l3_cache_id=0,
                    base_freq_mhz=0.0,
                    max_freq_mhz=0.0
                ))
        
        return sorted(cores, key=lambda c: c.core_id)
    
    def _detect_amd_topology(self, topology: CPUTopology):
        """
        检测AMD处理器拓扑
        
        Args:
            topology: CPU拓扑对象
        """
        ccd_info = self._amd_detector.detect_ccd_topology(
            topology.model_name, topology.cores
        )
        
        topology.ccd_count = ccd_info.get("ccd_count")
        topology.has_3d_vcache = ccd_info.get("has_3d_vcache", False)
    
    def _detect_intel_topology(self, topology: CPUTopology):
        """
        检测Intel处理器拓扑
        
        Args:
            topology: CPU拓扑对象
        """
        hybrid_info = self._intel_detector.detect_hybrid_topology(
            topology.model_name, topology.cores
        )
        
        topology.p_core_count = hybrid_info.get("p_core_count")
        topology.e_core_count = hybrid_info.get("e_core_count")
        topology.soc_core_count = hybrid_info.get("soc_core_count")
    
    def get_ccd0_cores(self) -> List[int]:
        """
        获取CCD0（大缓存）的核心ID列表
        
        Returns:
            CCD0核心ID列表
        """
        topology = self.detect()
        return self._amd_detector.get_ccd0_cores(topology.cores)
    
    def get_ccd1_cores(self) -> List[int]:
        """
        获取CCD1（高频率）的核心ID列表
        
        Returns:
            CCD1核心ID列表
        """
        topology = self.detect()
        return self._amd_detector.get_ccd1_cores(topology.cores)
    
    def get_p_cores(self) -> List[int]:
        """
        获取Intel P-Core列表
        
        Returns:
            P-Core核心ID列表
        """
        topology = self.detect()
        return self._intel_detector.get_p_cores(topology.cores)
    
    def get_e_cores(self) -> List[int]:
        """
        获取Intel E-Core列表
        
        Returns:
            E-Core核心ID列表
        """
        topology = self.detect()
        return self._intel_detector.get_e_cores(topology.cores)
    
    def get_soc_cores(self) -> List[int]:
        """
        获取Intel SoC Core列表
        
        Returns:
            SoC Core核心ID列表
        """
        topology = self.detect()
        return self._intel_detector.get_soc_cores(topology.cores)
    
    def get_recommended_affinity(self, task_type: str) -> Set[int]:
        """
        根据任务类型推荐CPU亲和度
        
        Args:
            task_type: "interactive" | "compute" | "background" | "cache_sensitive"
            
        Returns:
            推荐的核心ID集合
        """
        topology = self.detect()
        
        if topology.vendor == CPUVendor.AMD:
            return self._amd_detector.get_recommended_affinity(
                topology.cores, task_type
            )
        elif topology.vendor == CPUVendor.INTEL:
            return self._intel_detector.get_recommended_affinity(
                topology.cores, task_type
            )
        else:
            # 未知厂商，返回所有核心
            return set(c.core_id for c in topology.cores)
    
    def clear_cache(self):
        """清除拓扑缓存"""
        self._topology_cache = None
        self._cache_timestamp = 0
    
    def get_topology_summary(self) -> str:
        """
        获取拓扑摘要信息
        
        Returns:
            格式化的拓扑摘要字符串
        """
        topology = self.detect()
        
        lines = [
            f"CPU: {topology.model_name}",
            f"厂商: {topology.vendor.value}",
            f"物理核心: {topology.physical_cores}",
            f"逻辑核心: {topology.logical_cores}",
            f"NUMA节点: {topology.numa_nodes}",
            f"L3缓存: {topology.l3_caches}"
        ]
        
        if topology.vendor == CPUVendor.AMD:
            lines.extend([
                f"CCD数量: {topology.ccd_count}",
                f"3D V-Cache: {'是' if topology.has_3d_vcache else '否'}",
                f"CCD0核心: {self.get_ccd0_cores()}",
                f"CCD1核心: {self.get_ccd1_cores()}"
            ])
        elif topology.vendor == CPUVendor.INTEL:
            lines.extend([
                f"P-Core: {self.get_p_cores()} ({topology.p_core_count}个)",
                f"E-Core: {self.get_e_cores()} ({topology.e_core_count}个)",
            ])
            if topology.soc_core_count:
                lines.append(f"SoC Core: {self.get_soc_cores()} ({topology.soc_core_count}个)")
        
        return "\n".join(lines)


# 便捷函数
def detect_topology() -> CPUTopology:
    """
    检测CPU拓扑的便捷函数
    
    Returns:
        CPU拓扑信息
    """
    detector = CPUTopologyDetector()
    return detector.detect()


def get_detector() -> CPUTopologyDetector:
    """
    获取拓扑检测器实例
    
    Returns:
        CPUTopologyDetector实例
    """
    return CPUTopologyDetector()
