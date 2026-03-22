"""
进程监控与定期扫描系统

模块ID: Module-3E
模块名称: 进程监控与定期扫描系统

设计原则:
- 非持续监控，降低开销
- 缓存进程信息，减少重复查询
- 专注识别长时间运行任务
"""

import os
import re
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Set, Tuple, Any
from enum import Enum

import psutil

logger = logging.getLogger(__name__)


class ProcessType(Enum):
    """进程类型枚举"""
    SYSTEM = "system"
    USER_APP = "user_app"
    BACKGROUND = "background"
    TRAINING = "training"
    RENDERING = "rendering"
    COMPILATION = "compilation"
    SIMULATION = "simulation"
    UNKNOWN = "unknown"


@dataclass
class ProcessInfo:
    """进程信息数据类"""
    pid: int
    name: str
    cmdline: str
    create_time: datetime
    cpu_percent: float
    memory_percent: float
    runtime_minutes: float
    is_long_running: bool
    process_type: ProcessType
    protect_flag: bool  # 是否保护不关机
    username: Optional[str] = None
    memory_mb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "pid": self.pid,
            "name": self.name,
            "cmdline": self.cmdline[:200] if self.cmdline else "",  # 限制长度
            "create_time": self.create_time.isoformat(),
            "cpu_percent": round(self.cpu_percent, 2),
            "memory_percent": round(self.memory_percent, 2),
            "memory_mb": round(self.memory_mb, 2),
            "runtime_minutes": round(self.runtime_minutes, 2),
            "is_long_running": self.is_long_running,
            "process_type": self.process_type.value,
            "protect_flag": self.protect_flag,
            "username": self.username or "unknown",
        }


class ProcessScanner:
    """
    每10分钟扫描一次的进程监控器
    
    设计原则:
    - 非持续监控，降低开销
    - 缓存进程信息，减少重复查询
    - 专注识别长时间运行任务
    """
    
    SCAN_INTERVAL_MINUTES = 10
    LONG_RUNNING_THRESHOLD = 30  # 30分钟视为长时间运行
    
    # 系统进程白名单（始终保护）
    SYSTEM_WHITELIST: Set[str] = {
        'svchost.exe', 'csrss.exe', 'smss.exe', 'services.exe',
        'lsass.exe', 'winlogon.exe', 'explorer.exe', 'dwm.exe',
        'system', 'registry', 'memory compression', 'system interrupts',
        'idle', 'python.exe', 'python3.exe',  # 如果运行的是我们的节能系统
    }
    
    # 长时间运行任务特征（需要保护）
    LONG_RUNNING_PATTERNS: Dict[ProcessType, List[str]] = {
        ProcessType.TRAINING: ['train.py', 'main.py', 'python.exe', 'python3.exe', 
                               'train', 'fit.py', 'model.py', 'tensorflow', 'pytorch'],
        ProcessType.RENDERING: ['blender.exe', 'maya.exe', 'render.exe', '3dsmax.exe',
                                'cinema4d.exe', 'houdini.exe', 'vray.exe'],
        ProcessType.COMPILATION: ['cl.exe', 'gcc.exe', 'msbuild.exe', 'make.exe',
                                  'cmake.exe', 'ninja.exe', 'devenv.exe'],
        ProcessType.SIMULATION: ['ansys.exe', 'comsol.exe', 'matlab.exe', 'simulink.exe',
                                 'abaqus.exe', 'fluent.exe', 'starccm.exe'],
    }
    
    # 系统后台进程（通常需要保护）
    SYSTEM_BACKGROUND_PATTERNS: List[str] = [
        'sqlservr.exe', 'mysqld.exe', 'postgres.exe', 'mongod.exe',
        'redis-server.exe', 'nginx.exe', 'apache.exe', 'iis.exe',
        'docker.exe', 'containerd.exe', 'kubelet.exe',
    ]
    
    def __init__(self, 
                 long_running_threshold: int = 30,
                 scan_interval_minutes: int = 10,
                 custom_whitelist: Optional[Set[str]] = None):
        """
        初始化进程扫描器
        
        Args:
            long_running_threshold: 长时间运行阈值（分钟）
            scan_interval_minutes: 扫描间隔（分钟）
            custom_whitelist: 自定义白名单进程名集合
        """
        self._long_running_threshold = long_running_threshold
        self._scan_interval_minutes = scan_interval_minutes
        self._last_scan_time: Optional[datetime] = None
        self._process_cache: Dict[int, ProcessInfo] = {}
        self._long_running_tasks: List[ProcessInfo] = []
        self._protected_pids: Dict[int, str] = {}  # pid -> reason
        self._custom_whitelist = custom_whitelist or set()
        
        # 合并白名单
        self._all_whitelist = self.SYSTEM_WHITELIST | self._custom_whitelist
        
        logger.info(f"ProcessScanner initialized (threshold={long_running_threshold}min, "
                   f"interval={scan_interval_minutes}min)")
    
    def scan(self, force: bool = False) -> List[ProcessInfo]:
        """
        执行一次进程扫描
        
        Args:
            force: 是否强制扫描，忽略时间间隔
            
        Returns:
            当前运行的长时间任务列表
        """
        now = datetime.now()
        
        # 检查扫描间隔
        if not force and self._last_scan_time:
            elapsed = (now - self._last_scan_time).total_seconds() / 60
            if elapsed < self._scan_interval_minutes:
                logger.debug(f"Skipping scan, last scan was {elapsed:.1f} minutes ago")
                return self._long_running_tasks
        
        logger.info(f"Starting process scan at {now.isoformat()}")
        scan_start_time = time.time()
        
        try:
            current_processes: Dict[int, ProcessInfo] = {}
            long_running: List[ProcessInfo] = []
            
            # 遍历所有进程
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time',
                                              'cpu_percent', 'memory_percent', 'memory_info',
                                              'username']):
                try:
                    proc_info = self._analyze_process(proc)
                    if proc_info:
                        current_processes[proc_info.pid] = proc_info
                        if proc_info.is_long_running and proc_info.protect_flag:
                            long_running.append(proc_info)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
                except Exception as e:
                    logger.warning(f"Error analyzing process: {e}")
                    continue
            
            # 更新缓存
            self._process_cache = current_processes
            self._long_running_tasks = long_running
            self._last_scan_time = now
            
            scan_duration = time.time() - scan_start_time
            logger.info(f"Scan completed in {scan_duration*1000:.2f}ms, "
                       f"found {len(long_running)} long-running tasks")
            
            return long_running
            
        except Exception as e:
            logger.error(f"Error during process scan: {e}")
            return self._long_running_tasks
    
    def _analyze_process(self, proc: psutil.Process) -> Optional[ProcessInfo]:
        """
        分析单个进程
        
        Args:
            proc: psutil.Process对象
            
        Returns:
            ProcessInfo对象，如果无法分析则返回None
        """
        try:
            pinfo = proc.info
            pid = pinfo.get('pid', 0)
            name = pinfo.get('name', '') or ''
            cmdline_list = pinfo.get('cmdline', [])
            cmdline = ' '.join(cmdline_list) if cmdline_list else name
            
            # 获取创建时间
            create_timestamp = pinfo.get('create_time', 0)
            if create_timestamp:
                create_time = datetime.fromtimestamp(create_timestamp)
            else:
                create_time = datetime.now()
            
            # 计算运行时间
            runtime_minutes = (datetime.now() - create_time).total_seconds() / 60
            
            # 判断进程类型
            process_type = self._classify_process(name, cmdline)
            
            # 判断是否长时间运行
            is_long_running = runtime_minutes >= self._long_running_threshold
            
            # 判断是否需要保护
            protect_flag = self._should_protect(name, cmdline, process_type, pid)
            
            # 获取资源使用
            cpu_percent = pinfo.get('cpu_percent', 0.0) or 0.0
            memory_percent = pinfo.get('memory_percent', 0.0) or 0.0
            memory_info = pinfo.get('memory_info', None)
            memory_mb = (memory_info.rss / 1024 / 1024) if memory_info else 0.0
            
            # 获取用户名
            username = pinfo.get('username', None)
            
            return ProcessInfo(
                pid=pid,
                name=name,
                cmdline=cmdline,
                create_time=create_time,
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                runtime_minutes=runtime_minutes,
                is_long_running=is_long_running,
                process_type=process_type,
                protect_flag=protect_flag,
                username=username,
                memory_mb=memory_mb,
            )
            
        except Exception as e:
            logger.debug(f"Error analyzing process {proc.pid}: {e}")
            return None
    
    def _classify_process(self, name: str, cmdline: str) -> ProcessType:
        """
        根据进程名和命令行分类进程类型
        
        分类优先级:
        1. 检查长时间运行任务模式（训练、渲染等）
        2. 检查系统后台进程
        3. 检查系统白名单
        4. 默认为用户应用
        
        Args:
            name: 进程名
            cmdline: 命令行
            
        Returns:
            进程类型枚举
        """
        name_lower = name.lower()
        cmdline_lower = cmdline.lower()
        
        # 首先检查长时间运行模式（优先级高于白名单）
        # 这样即使python.exe在白名单中，如果运行的是训练脚本，也会被正确分类
        for proc_type, patterns in self.LONG_RUNNING_PATTERNS.items():
            for pattern in patterns:
                pattern_lower = pattern.lower()
                # 对于python.exe，需要检查命令行参数来判断具体类型
                if name_lower in ('python.exe', 'python3.exe', 'python', 'python3'):
                    # 检查命令行中是否包含特定模式
                    if pattern_lower in cmdline_lower and pattern_lower not in ('python.exe', 'python3.exe'):
                        return proc_type
                else:
                    # 非python进程直接匹配名称或命令行
                    if pattern_lower in name_lower or pattern_lower in cmdline_lower:
                        return proc_type
        
        # 检查系统后台进程
        for pattern in self.SYSTEM_BACKGROUND_PATTERNS:
            if pattern.lower() in name_lower:
                return ProcessType.BACKGROUND
        
        # 检查是否在白名单中（系统进程）
        if name_lower in {n.lower() for n in self._all_whitelist}:
            return ProcessType.SYSTEM
        
        # 默认为用户应用
        return ProcessType.USER_APP
    
    def _should_protect(self, name: str, cmdline: str, 
                        process_type: ProcessType, pid: int) -> bool:
        """
        判断进程是否应该被保护（阻止关机）
        
        Args:
            name: 进程名
            cmdline: 命令行
            process_type: 进程类型
            pid: 进程ID
            
        Returns:
            是否应该保护
        """
        # 检查是否被手动标记为保护
        if pid in self._protected_pids:
            return True
        
        name_lower = name.lower()
        
        # 系统进程始终保护
        if process_type == ProcessType.SYSTEM:
            return True
        
        # 白名单进程保护
        if name_lower in {n.lower() for n in self._all_whitelist}:
            return True
        
        # 长时间运行的特殊任务类型需要保护
        if process_type in (ProcessType.TRAINING, ProcessType.RENDERING,
                           ProcessType.COMPILATION, ProcessType.SIMULATION):
            return True
        
        # 系统后台服务保护
        for pattern in self.SYSTEM_BACKGROUND_PATTERNS:
            if pattern.lower() in name_lower:
                return True
        
        return False
    
    def get_long_running_tasks(self) -> List[ProcessInfo]:
        """
        获取缓存的长时间运行任务
        
        Returns:
            长时间运行任务列表
        """
        return self._long_running_tasks.copy()
    
    def get_all_processes(self) -> Dict[int, ProcessInfo]:
        """
        获取所有缓存的进程信息
        
        Returns:
            进程ID到ProcessInfo的映射
        """
        return self._process_cache.copy()
    
    def should_prevent_shutdown(self) -> Tuple[bool, List[str]]:
        """
        判断是否应阻止关机
        
        Returns:
            (是否阻止, 阻止原因列表)
        """
        # 确保数据是最新的
        if not self._last_scan_time or \
           (datetime.now() - self._last_scan_time).total_seconds() > 600:
            self.scan()
        
        reasons: List[str] = []
        
        for task in self._long_running_tasks:
            if task.protect_flag:
                reason = (f"长时间运行任务: {task.name} (PID: {task.pid}, "
                         f"运行时间: {task.runtime_minutes:.1f}分钟, "
                         f"类型: {task.process_type.value})")
                reasons.append(reason)
        
        return len(reasons) > 0, reasons
    
    def mark_protected(self, pid: int, reason: str) -> bool:
        """
        手动标记进程为保护状态
        
        Args:
            pid: 进程ID
            reason: 保护原因
            
        Returns:
            是否成功标记
        """
        try:
            # 验证进程是否存在
            if not psutil.pid_exists(pid):
                logger.warning(f"Cannot mark PID {pid} as protected: process does not exist")
                return False
            
            self._protected_pids[pid] = reason
            logger.info(f"Marked PID {pid} as protected: {reason}")
            
            # 更新缓存
            if pid in self._process_cache:
                self._process_cache[pid].protect_flag = True
            
            return True
            
        except Exception as e:
            logger.error(f"Error marking PID {pid} as protected: {e}")
            return False
    
    def unmark_protected(self, pid: int) -> bool:
        """
        取消进程的保护状态
        
        Args:
            pid: 进程ID
            
        Returns:
            是否成功取消标记
        """
        if pid in self._protected_pids:
            del self._protected_pids[pid]
            logger.info(f"Removed protected status from PID {pid}")
            
            # 更新缓存
            if pid in self._process_cache:
                self._process_cache[pid].protect_flag = False
            
            return True
        return False
    
    def get_protected_processes(self) -> List[ProcessInfo]:
        """
        获取所有被标记为保护的进程
        
        Returns:
            受保护进程列表
        """
        result = []
        for pid, reason in self._protected_pids.items():
            if pid in self._process_cache:
                result.append(self._process_cache[pid])
        return result
    
    def get_scan_stats(self) -> Dict[str, Any]:
        """
        获取扫描统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "last_scan_time": self._last_scan_time.isoformat() if self._last_scan_time else None,
            "total_processes": len(self._process_cache),
            "long_running_tasks": len(self._long_running_tasks),
            "protected_processes": len(self._protected_pids),
            "scan_interval_minutes": self._scan_interval_minutes,
            "long_running_threshold": self._long_running_threshold,
        }
    
    def add_to_whitelist(self, process_name: str) -> None:
        """
        添加进程到白名单
        
        Args:
            process_name: 进程名
        """
        self._custom_whitelist.add(process_name)
        self._all_whitelist = self.SYSTEM_WHITELIST | self._custom_whitelist
        logger.info(f"Added {process_name} to whitelist")
    
    def remove_from_whitelist(self, process_name: str) -> None:
        """
        从白名单中移除进程
        
        Args:
            process_name: 进程名
        """
        self._custom_whitelist.discard(process_name)
        self._all_whitelist = self.SYSTEM_WHITELIST | self._custom_whitelist
        logger.info(f"Removed {process_name} from whitelist")


# 全局扫描器实例（单例模式）
_scanner_instance: Optional[ProcessScanner] = None


def get_scanner(**kwargs) -> ProcessScanner:
    """
    获取全局扫描器实例（单例模式）
    
    Args:
        **kwargs: 传递给ProcessScanner构造函数的参数
        
    Returns:
        ProcessScanner实例
    """
    global _scanner_instance
    if _scanner_instance is None:
        _scanner_instance = ProcessScanner(**kwargs)
    return _scanner_instance


def reset_scanner() -> None:
    """重置全局扫描器实例"""
    global _scanner_instance
    _scanner_instance = None
