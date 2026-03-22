"""
CPU亲和度类型定义模块

定义CPU亲和度管理所需的数据结构和枚举类型
"""

from dataclasses import dataclass
from typing import Set, List, Optional
from enum import Enum


class TaskType(Enum):
    """任务类型"""
    INTERACTIVE = "interactive"       # 交互式应用（前台）
    COMPUTE = "compute"               # 计算密集型
    CACHE_SENSITIVE = "cache"         # 缓存敏感型（游戏）
    BACKGROUND = "background"         # 后台任务
    SYSTEM = "system"                 # 系统进程


@dataclass
class AffinityPolicy:
    """亲和度策略"""
    task_type: TaskType
    preferred_cores: Set[int]         # 首选核心
    allowed_cores: Set[int]           # 允许运行的核心
    excluded_cores: Set[int]          # 排除的核心
    priority_boost: bool = False      # 是否提升优先级


@dataclass
class CPUSetInfo:
    """CPU Set信息"""
    id: int                           # CPU Set ID
    group: int                        # 处理器组
    logical_processor_index: int      # 逻辑处理器索引
    core_index: int                   # 核心索引
    numa_node_index: int              # NUMA节点索引
    efficiency_class: int             # 效率等级 (Intel大小核)
    all_flags: int                    # 所有标志


@dataclass
class ThreadAffinityInfo:
    """线程亲和度信息"""
    thread_id: int                    # 线程ID
    thread_handle: int                # 线程句柄
    current_mask: int                 # 当前亲和度掩码
    current_group: int                # 当前处理器组
    cpu_set_ids: Optional[List[int]] = None  # 当前CPU Set IDs


@dataclass
class ProcessAffinityInfo:
    """进程亲和度信息"""
    pid: int                          # 进程ID
    process_handle: int               # 进程句柄
    default_cpu_sets: Optional[List[int]] = None  # 默认CPU Sets
    current_mask: Optional[int] = None  # 当前亲和度掩码
