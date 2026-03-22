"""
CPU亲和度管理器模块

使用Windows CPU Sets API和Thread Affinity API
实现进程和线程级别的亲和度控制
支持AMD CCD和Intel大小核的智能调度
"""

import logging
from typing import List, Set, Optional, Dict, Tuple
import os

try:
    from .affinity_types import TaskType, AffinityPolicy, CPUSetInfo, ProcessAffinityInfo
    from .affinity_api import WindowsAffinityAPI, get_affinity_api
except ImportError:
    from affinity_types import TaskType, AffinityPolicy, CPUSetInfo, ProcessAffinityInfo
    from affinity_api import WindowsAffinityAPI, get_affinity_api

try:
    from .topology_types import CPUTopology, CPUVendor, CoreType, CCDType, TaskType as TopoTaskType
    from .topology import CPUTopologyDetector
except ImportError:
    from topology_types import CPUTopology, CPUVendor, CoreType, CCDType
    from topology import CPUTopologyDetector

# 设置日志
logger = logging.getLogger(__name__)


class CPUAffinityManager:
    """
    CPU亲和度管理器
    
    使用Windows CPU Sets API和Thread Affinity API
    实现进程和线程级别的亲和度控制
    
    支持:
    - Intel大小核架构的智能调度
    - AMD CCD架构的智能调度
    - 后台任务隔离到能效核心
    - 交互式应用性能优化
    """
    
    def __init__(self, topology_detector: Optional[CPUTopologyDetector] = None):
        """
        初始化CPU亲和度管理器
        
        Args:
            topology_detector: CPUTopologyDetector实例，如果为None则自动创建
        """
        self._api = get_affinity_api()
        self._topology_detector = topology_detector or CPUTopologyDetector()
        self._topology: Optional[CPUTopology] = None
        self._cpu_set_info: List[Dict] = []
        self._core_to_cpu_set: Dict[int, int] = {}  # 逻辑核心ID -> CPU Set ID
        
        # 缓存推荐的核心分配
        self._recommended_cores: Dict[TaskType, Set[int]] = {}
        
        self._initialize()
    
    def _initialize(self):
        """初始化拓扑信息和CPU Set映射"""
        try:
            # 检测CPU拓扑
            self._topology = self._topology_detector.detect()
            
            # 获取CPU Set信息
            self._cpu_set_info = self._api.get_system_cpu_set_information()
            
            # 构建核心到CPU Set的映射
            self._build_core_mapping()
            
            # 预计算推荐的核心分配
            self._calculate_recommended_cores()
            
            logger.info(f"CPU亲和度管理器初始化完成: {self._topology.model_name}")
            logger.info(f"CPU Sets API支持: {self._api.cpu_sets_supported}")
            logger.info(f"CPU Set数量: {len(self._cpu_set_info)}")
            
        except Exception as e:
            logger.error(f"初始化CPU亲和度管理器失败: {e}")
            self._topology = None
    
    def _build_core_mapping(self):
        """构建逻辑核心到CPU Set的映射"""
        if not self._cpu_set_info:
            return
        
        for cpu_set in self._cpu_set_info:
            logical_index = cpu_set.get('LogicalProcessorIndex', 0)
            cpu_set_id = cpu_set.get('Id', 0)
            self._core_to_cpu_set[logical_index] = cpu_set_id
        
        logger.debug(f"核心到CPU Set映射: {self._core_to_cpu_set}")
    
    def _calculate_recommended_cores(self):
        """计算各种任务类型推荐的核心"""
        if self._topology is None:
            return
        
        vendor = self._topology.vendor
        cores = self._topology.cores
        
        if vendor == CPUVendor.INTEL:
            # Intel大小核架构
            self._calculate_intel_recommended_cores(cores)
        elif vendor == CPUVendor.AMD:
            # AMD CCD架构
            self._calculate_amd_recommended_cores(cores)
        else:
            # 通用架构
            self._calculate_generic_recommended_cores(cores)
    
    def _calculate_intel_recommended_cores(self, cores: List):
        """计算Intel大小核推荐核心"""
        # P-Core (性能核) - 适合交互式应用和缓存敏感任务
        p_cores = {c.logical_processor_id for c in cores if c.core_type == CoreType.P_CORE}
        
        # E-Core (能效核) - 适合后台任务
        e_cores = {c.logical_processor_id for c in cores if c.core_type == CoreType.E_CORE}
        
        # SoC Core - 适合低优先级后台任务
        soc_cores = {c.logical_processor_id for c in cores if c.core_type == CoreType.SOC_CORE}
        
        # 所有核心
        all_cores = {c.logical_processor_id for c in cores}
        
        self._recommended_cores[TaskType.INTERACTIVE] = p_cores
        self._recommended_cores[TaskType.CACHE_SENSITIVE] = p_cores
        self._recommended_cores[TaskType.COMPUTE] = all_cores
        self._recommended_cores[TaskType.BACKGROUND] = e_cores | soc_cores
        self._recommended_cores[TaskType.SYSTEM] = all_cores
        
        logger.debug(f"Intel推荐核心 - P-Core: {p_cores}, E-Core: {e_cores}, SoC: {soc_cores}")
    
    def _calculate_amd_recommended_cores(self, cores: List):
        """计算AMD CCD推荐核心"""
        # CCD0 (通常有3D V-Cache) - 适合缓存敏感任务
        ccd0_cores = {c.logical_processor_id for c in cores if c.ccd_type == CCDType.CCD0_CACHE}
        
        # CCD1 (通常频率更高) - 适合计算密集型任务
        ccd1_cores = {c.logical_processor_id for c in cores if c.ccd_type == CCDType.CCD1_FREQ}
        
        # 所有核心
        all_cores = {c.logical_processor_id for c in cores}
        
        # 后台任务使用CCD1（非3D V-Cache CCD）
        background_cores = ccd1_cores if ccd1_cores else all_cores
        
        self._recommended_cores[TaskType.INTERACTIVE] = ccd0_cores if ccd0_cores else all_cores
        self._recommended_cores[TaskType.CACHE_SENSITIVE] = ccd0_cores if ccd0_cores else all_cores
        self._recommended_cores[TaskType.COMPUTE] = all_cores
        self._recommended_cores[TaskType.BACKGROUND] = background_cores
        self._recommended_cores[TaskType.SYSTEM] = all_cores
        
        logger.debug(f"AMD推荐核心 - CCD0: {ccd0_cores}, CCD1: {ccd1_cores}")
    
    def _calculate_generic_recommended_cores(self, cores: List):
        """计算通用架构推荐核心"""
        all_cores = {c.logical_processor_id for c in cores}
        
        # 对于通用架构，所有任务类型使用所有核心
        for task_type in TaskType:
            self._recommended_cores[task_type] = all_cores
        
        logger.debug(f"通用架构推荐核心: {all_cores}")
    
    def _cores_to_cpu_sets(self, cores: Set[int]) -> List[int]:
        """
        将逻辑核心ID转换为CPU Set ID
        
        Args:
            cores: 逻辑核心ID集合
            
        Returns:
            CPU Set ID列表
        """
        if not self._cpu_set_info:
            # 如果不支持CPU Sets，返回空列表
            return []
        
        cpu_sets = []
        for core_id in cores:
            if core_id in self._core_to_cpu_set:
                cpu_sets.append(self._core_to_cpu_set[core_id])
        
        return cpu_sets
    
    def _cores_to_mask(self, cores: Set[int]) -> int:
        """
        将逻辑核心ID转换为亲和度掩码
        
        Args:
            cores: 逻辑核心ID集合
            
        Returns:
            亲和度位掩码
        """
        mask = 0
        for core_id in cores:
            mask |= (1 << core_id)
        return mask
    
    def set_process_affinity(self, pid: int, cpu_set_ids: List[int]) -> bool:
        """
        设置进程默认CPU Sets
        
        影响进程内所有未单独设置亲和度的线程
        
        Args:
            pid: 进程ID
            cpu_set_ids: CPU Set ID列表
            
        Returns:
            是否设置成功
        """
        if not cpu_set_ids:
            logger.warning("CPU Set ID列表为空")
            return False
        
        # 优先使用CPU Sets API
        if self._api.cpu_sets_supported:
            return self._api.set_process_default_cpu_sets(pid, cpu_set_ids)
        else:
            # 回退到传统Affinity Mask
            logger.warning("CPU Sets API不受支持，使用Affinity Mask回退")
            # 从CPU Set IDs推断核心ID
            cores = set()
            for cpu_set in self._cpu_set_info:
                if cpu_set['Id'] in cpu_set_ids:
                    cores.add(cpu_set['LogicalProcessorIndex'])
            
            if cores:
                mask = self._cores_to_mask(cores)
                return self._api.set_process_affinity_mask(pid, mask)
            return False
    
    def set_thread_affinity(self, thread_id: int, cpu_set_ids: List[int]) -> bool:
        """
        设置特定线程的CPU Sets
        
        Args:
            thread_id: 线程ID
            cpu_set_ids: CPU Set ID列表
            
        Returns:
            是否设置成功
        """
        if not cpu_set_ids:
            logger.warning("CPU Set ID列表为空")
            return False
        
        # 优先使用CPU Sets API
        if self._api.cpu_sets_supported:
            return self._api.set_thread_selected_cpu_sets(thread_id, cpu_set_ids)
        else:
            # 回退到传统Affinity Mask
            logger.warning("CPU Sets API不受支持，使用Affinity Mask回退")
            cores = set()
            for cpu_set in self._cpu_set_info:
                if cpu_set['Id'] in cpu_set_ids:
                    cores.add(cpu_set['LogicalProcessorIndex'])
            
            if cores:
                mask = self._cores_to_mask(cores)
                return self._api.set_thread_affinity_mask(thread_id, mask) is not None
            return False
    
    def set_thread_affinity_mask(self, thread_handle: int, mask: int) -> bool:
        """
        使用传统Affinity Mask设置线程亲和度
        
        用于旧版Windows或不支持CPU Sets的情况
        
        Args:
            thread_handle: 线程ID（不是句柄，函数内部会打开线程）
            mask: 亲和度位掩码
            
        Returns:
            是否设置成功
        """
        result = self._api.set_thread_affinity_mask(thread_handle, mask)
        return result is not None
    
    def isolate_background_process(self, pid: int) -> bool:
        """
        将后台进程隔离到能效核心
        
        - Intel: 隔离到E-Core或SoC Core
        - AMD: 隔离到CCD1（非3D V-Cache CCD）
        
        Args:
            pid: 进程ID
            
        Returns:
            是否设置成功
        """
        if self._topology is None:
            logger.error("CPU拓扑信息不可用")
            return False
        
        # 获取后台任务推荐核心
        background_cores = self._recommended_cores.get(TaskType.BACKGROUND, set())
        
        if not background_cores:
            logger.warning("没有可用的后台任务核心")
            return False
        
        logger.info(f"隔离后台进程 {pid} 到核心: {background_cores}")
        
        # 转换为CPU Sets
        cpu_sets = self._cores_to_cpu_sets(background_cores)
        
        if cpu_sets and self._api.cpu_sets_supported:
            return self.set_process_affinity(pid, cpu_sets)
        else:
            # 使用Affinity Mask
            mask = self._cores_to_mask(background_cores)
            return self._api.set_process_affinity_mask(pid, mask)
    
    def optimize_for_interactive(self, pid: int) -> bool:
        """
        为交互式应用优化亲和度
        
        - Intel: 分配到P-Core
        - AMD: 分配到CCD0（3D V-Cache）
        
        Args:
            pid: 进程ID
            
        Returns:
            是否设置成功
        """
        if self._topology is None:
            logger.error("CPU拓扑信息不可用")
            return False
        
        # 获取交互式任务推荐核心
        interactive_cores = self._recommended_cores.get(TaskType.INTERACTIVE, set())
        
        if not interactive_cores:
            logger.warning("没有可用的交互式任务核心")
            return False
        
        logger.info(f"优化交互式进程 {pid} 到核心: {interactive_cores}")
        
        # 转换为CPU Sets
        cpu_sets = self._cores_to_cpu_sets(interactive_cores)
        
        if cpu_sets and self._api.cpu_sets_supported:
            return self.set_process_affinity(pid, cpu_sets)
        else:
            # 使用Affinity Mask
            mask = self._cores_to_mask(interactive_cores)
            return self._api.set_process_affinity_mask(pid, mask)
    
    def optimize_for_cache_sensitive(self, pid: int) -> bool:
        """
        为缓存敏感型应用优化亲和度（如游戏）
        
        - Intel: 分配到P-Core
        - AMD: 分配到CCD0（3D V-Cache）
        
        Args:
            pid: 进程ID
            
        Returns:
            是否设置成功
        """
        if self._topology is None:
            logger.error("CPU拓扑信息不可用")
            return False
        
        # 获取缓存敏感任务推荐核心
        cache_cores = self._recommended_cores.get(TaskType.CACHE_SENSITIVE, set())
        
        if not cache_cores:
            logger.warning("没有可用的缓存敏感任务核心")
            return False
        
        logger.info(f"优化缓存敏感进程 {pid} 到核心: {cache_cores}")
        
        # 转换为CPU Sets
        cpu_sets = self._cores_to_cpu_sets(cache_cores)
        
        if cpu_sets and self._api.cpu_sets_supported:
            return self.set_process_affinity(pid, cpu_sets)
        else:
            # 使用Affinity Mask
            mask = self._cores_to_mask(cache_cores)
            return self._api.set_process_affinity_mask(pid, mask)
    
    def optimize_for_compute(self, pid: int) -> bool:
        """
        为计算密集型应用优化亲和度
        
        使用所有可用核心
        
        Args:
            pid: 进程ID
            
        Returns:
            是否设置成功
        """
        if self._topology is None:
            logger.error("CPU拓扑信息不可用")
            return False
        
        # 获取计算任务推荐核心（所有核心）
        compute_cores = self._recommended_cores.get(TaskType.COMPUTE, set())
        
        if not compute_cores:
            logger.warning("没有可用的计算任务核心")
            return False
        
        logger.info(f"优化计算密集型进程 {pid} 到所有核心: {compute_cores}")
        
        # 转换为CPU Sets
        cpu_sets = self._cores_to_cpu_sets(compute_cores)
        
        if cpu_sets and self._api.cpu_sets_supported:
            return self.set_process_affinity(pid, cpu_sets)
        else:
            # 使用Affinity Mask
            mask = self._cores_to_mask(compute_cores)
            return self._api.set_process_affinity_mask(pid, mask)
    
    def apply_policy(self, pid: int, policy: AffinityPolicy) -> bool:
        """
        应用亲和度策略
        
        Args:
            pid: 进程ID
            policy: 亲和度策略
            
        Returns:
            是否设置成功
        """
        if self._topology is None:
            logger.error("CPU拓扑信息不可用")
            return False
        
        # 确定最终使用的核心集合
        if policy.preferred_cores:
            # 使用首选核心
            target_cores = policy.preferred_cores
        elif policy.allowed_cores:
            # 使用允许的核心
            target_cores = policy.allowed_cores
        else:
            # 使用任务类型推荐的核心
            target_cores = self._recommended_cores.get(policy.task_type, set())
        
        # 排除指定核心
        if policy.excluded_cores:
            target_cores = target_cores - policy.excluded_cores
        
        if not target_cores:
            logger.warning("没有可用的目标核心")
            return False
        
        logger.info(f"应用策略到进程 {pid}: 任务类型={policy.task_type.value}, 核心={target_cores}")
        
        # 转换为CPU Sets
        cpu_sets = self._cores_to_cpu_sets(target_cores)
        
        success = False
        if cpu_sets and self._api.cpu_sets_supported:
            success = self.set_process_affinity(pid, cpu_sets)
        else:
            # 使用Affinity Mask
            mask = self._cores_to_mask(target_cores)
            success = self._api.set_process_affinity_mask(pid, mask)
        
        # 如果需要提升优先级
        if success and policy.priority_boost:
            # 注意：这里只设置亲和度，优先级提升需要额外的API调用
            logger.info(f"进程 {pid} 的亲和度已设置，优先级提升需要额外实现")
        
        return success
    
    def reset_affinity(self, pid: int) -> bool:
        """
        重置进程亲和度为系统默认
        
        Args:
            pid: 进程ID
            
        Returns:
            是否重置成功
        """
        if self._topology is None:
            logger.error("CPU拓扑信息不可用")
            return False
        
        # 获取所有核心
        all_cores = self._recommended_cores.get(TaskType.SYSTEM, set())
        
        if not all_cores:
            # 使用系统信息获取所有处理器
            sys_info = self._api.get_system_info()
            num_processors = sys_info.get('number_of_processors', os.cpu_count() or 1)
            all_cores = set(range(num_processors))
        
        logger.info(f"重置进程 {pid} 的亲和度到所有核心: {all_cores}")
        
        # 转换为CPU Sets
        cpu_sets = self._cores_to_cpu_sets(all_cores)
        
        if cpu_sets and self._api.cpu_sets_supported:
            return self.set_process_affinity(pid, cpu_sets)
        else:
            # 使用Affinity Mask
            mask = self._cores_to_mask(all_cores)
            return self._api.set_process_affinity_mask(pid, mask)
    
    def get_process_affinity_info(self, pid: int) -> Optional[ProcessAffinityInfo]:
        """
        获取进程亲和度信息
        
        Args:
            pid: 进程ID
            
        Returns:
            进程亲和度信息，失败返回None
        """
        info = ProcessAffinityInfo(pid=pid, process_handle=0)
        
        # 获取CPU Sets
        if self._api.cpu_sets_supported:
            info.default_cpu_sets = self._api.get_process_default_cpu_sets(pid)
        
        # 获取Affinity Mask
        mask_result = self._api.get_process_affinity_mask(pid)
        if mask_result:
            info.current_mask = mask_result[0]
        
        return info
    
    def get_recommended_cores(self, task_type: TaskType) -> Set[int]:
        """
        获取指定任务类型推荐的核心
        
        Args:
            task_type: 任务类型
            
        Returns:
            推荐的核心ID集合
        """
        return self._recommended_cores.get(task_type, set())
    
    def get_topology(self) -> Optional[CPUTopology]:
        """
        获取CPU拓扑信息
        
        Returns:
            CPU拓扑信息，如果不可用返回None
        """
        return self._topology
    
    def get_cpu_set_info(self) -> List[Dict]:
        """
        获取CPU Set信息
        
        Returns:
            CPU Set信息列表
        """
        return self._cpu_set_info
    
    def is_cpu_sets_supported(self) -> bool:
        """
        检查是否支持CPU Sets API
        
        Returns:
            是否支持CPU Sets API
        """
        return self._api.cpu_sets_supported
    
    def refresh_topology(self):
        """刷新CPU拓扑信息"""
        self._initialize()


# 创建全局管理器实例
_affinity_manager = None

def get_affinity_manager(topology_detector: Optional[CPUTopologyDetector] = None) -> CPUAffinityManager:
    """
    获取全局CPUAffinityManager实例
    
    Args:
        topology_detector: 可选的CPUTopologyDetector实例
        
    Returns:
        CPUAffinityManager实例
    """
    global _affinity_manager
    if _affinity_manager is None:
        _affinity_manager = CPUAffinityManager(topology_detector)
    return _affinity_manager
