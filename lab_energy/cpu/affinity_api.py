"""
Windows亲和度API底层封装模块

封装Windows CPU Sets API和Thread Affinity API
支持Windows 10 1703+的CPU Sets和传统Affinity Mask
"""

import ctypes
from ctypes import wintypes
from typing import List, Optional, Dict, Tuple
import logging
import platform

# 设置日志
logger = logging.getLogger(__name__)

# Windows API常量
ERROR_INSUFFICIENT_BUFFER = 122
ERROR_INVALID_PARAMETER = 87
ERROR_ACCESS_DENIED = 5
ERROR_NOT_SUPPORTED = 50

# 处理器访问权限
PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_SET_INFORMATION = 0x0200
PROCESS_ALL_ACCESS = 0x1F0FFF

# 线程访问权限
THREAD_QUERY_INFORMATION = 0x0040
THREAD_SET_INFORMATION = 0x0020
THREAD_SET_LIMITED_INFORMATION = 0x0400
THREAD_ALL_ACCESS = 0x1F03FF

# 定义ULONG_PTR类型
if ctypes.sizeof(ctypes.c_void_p) == 8:
    ULONG_PTR = ctypes.c_ulonglong
else:
    ULONG_PTR = ctypes.c_ulong


class GROUP_AFFINITY(ctypes.Structure):
    """处理器组亲和度结构"""
    _fields_ = [
        ("Mask", ULONG_PTR),
        ("Group", wintypes.WORD),
        ("Reserved", wintypes.WORD * 3),
    ]


class SYSTEM_CPU_SET_INFORMATION(ctypes.Structure):
    """系统CPU Set信息结构"""
    _fields_ = [
        ("Id", wintypes.DWORD),
        ("Group", wintypes.WORD),
        ("LogicalProcessorIndex", wintypes.BYTE),
        ("CoreIndex", wintypes.BYTE),
        ("LastLevelCacheIndex", wintypes.BYTE),
        ("NumaNodeIndex", wintypes.BYTE),
        ("EfficiencyClass", wintypes.BYTE),
        ("AllFlags", wintypes.BYTE),
        ("Reserved", wintypes.BYTE),
        ("AllocationTag", ULONG_PTR),
    ]


class WindowsAffinityAPI:
    """Windows亲和度API底层封装"""
    
    def __init__(self):
        self.kernel32 = ctypes.windll.kernel32
        self._cpu_sets_supported = self._check_cpu_sets_support()
        
    def _check_cpu_sets_support(self) -> bool:
        """检查系统是否支持CPU Sets API"""
        try:
            # Windows 10 1703+ 支持CPU Sets
            # 检查kernel32中是否存在SetProcessDefaultCpuSets函数
            if not hasattr(self.kernel32, 'SetProcessDefaultCpuSets'):
                return False
            
            # 尝试调用GetSystemCpuSetInformation获取信息
            buffer_size = wintypes.DWORD(0)
            result = self.kernel32.GetSystemCpuSetInformation(
                None, 0, ctypes.byref(buffer_size), None, 0
            )
            # 如果返回ERROR_INSUFFICIENT_BUFFER，说明API可用
            return result == 0 or ctypes.get_last_error() == ERROR_INSUFFICIENT_BUFFER
        except Exception as e:
            logger.debug(f"CPU Sets API检查失败: {e}")
            return False
    
    @property
    def cpu_sets_supported(self) -> bool:
        """是否支持CPU Sets API"""
        return self._cpu_sets_supported
    
    def _open_process(self, pid: int, access: int = PROCESS_SET_INFORMATION) -> Optional[int]:
        """
        打开进程获取句柄
        
        Args:
            pid: 进程ID
            access: 访问权限
            
        Returns:
            进程句柄，失败返回None
        """
        try:
            handle = self.kernel32.OpenProcess(access, False, pid)
            if handle == 0:
                error = ctypes.get_last_error()
                logger.warning(f"无法打开进程 {pid}: 错误码 {error}")
                return None
            return handle
        except Exception as e:
            logger.error(f"打开进程 {pid} 失败: {e}")
            return None
    
    def _open_thread(self, tid: int, access: int = THREAD_SET_INFORMATION) -> Optional[int]:
        """
        打开线程获取句柄
        
        Args:
            tid: 线程ID
            access: 访问权限
            
        Returns:
            线程句柄，失败返回None
        """
        try:
            handle = self.kernel32.OpenThread(access, False, tid)
            if handle == 0:
                error = ctypes.get_last_error()
                logger.warning(f"无法打开线程 {tid}: 错误码 {error}")
                return None
            return handle
        except Exception as e:
            logger.error(f"打开线程 {tid} 失败: {e}")
            return None
    
    def _close_handle(self, handle: int) -> bool:
        """关闭句柄"""
        try:
            return bool(self.kernel32.CloseHandle(handle))
        except Exception:
            return False
    
    # ========== CPU Sets API (Windows 10 1703+) ==========
    
    def get_system_cpu_set_information(self) -> List[Dict]:
        """
        获取系统中所有CPU Sets的信息
        
        Returns:
            CPU Set信息列表，每个字典包含:
            - Id: CPU Set ID
            - Group: 处理器组
            - LogicalProcessorIndex: 逻辑处理器索引
            - CoreIndex: 核心索引
            - NumaNodeIndex: NUMA节点索引
            - EfficiencyClass: 效率等级 (Intel大小核)
            - AllFlags: 所有标志
        """
        if not self._cpu_sets_supported:
            logger.warning("CPU Sets API不受支持")
            return []
        
        try:
            # 获取所需缓冲区大小
            buffer_size = wintypes.DWORD(0)
            result = self.kernel32.GetSystemCpuSetInformation(
                None, 0, ctypes.byref(buffer_size), None, 0
            )
            
            if result == 0 and ctypes.get_last_error() != ERROR_INSUFFICIENT_BUFFER:
                logger.error("GetSystemCpuSetInformation 获取缓冲区大小失败")
                return []
            
            # 分配缓冲区
            buffer = (ctypes.c_byte * buffer_size.value)()
            
            # 获取CPU Set信息
            result = self.kernel32.GetSystemCpuSetInformation(
                buffer, buffer_size, ctypes.byref(buffer_size), None, 0
            )
            
            if result == 0:
                logger.error("GetSystemCpuSetInformation 获取信息失败")
                return []
            
            # 解析结果
            cpu_sets = []
            offset = 0
            info_size = ctypes.sizeof(SYSTEM_CPU_SET_INFORMATION)
            
            while offset + info_size <= buffer_size.value:
                info = SYSTEM_CPU_SET_INFORMATION.from_buffer(buffer, offset)
                cpu_sets.append({
                    'Id': info.Id,
                    'Group': info.Group,
                    'LogicalProcessorIndex': info.LogicalProcessorIndex,
                    'CoreIndex': info.CoreIndex,
                    'LastLevelCacheIndex': info.LastLevelCacheIndex,
                    'NumaNodeIndex': info.NumaNodeIndex,
                    'EfficiencyClass': info.EfficiencyClass,
                    'AllFlags': info.AllFlags,
                    'AllocationTag': info.AllocationTag,
                })
                offset += info_size
            
            return cpu_sets
            
        except Exception as e:
            logger.error(f"获取CPU Set信息失败: {e}")
            return []
    
    def set_process_default_cpu_sets(self, pid: int, cpu_set_ids: List[int]) -> bool:
        """
        设置进程默认CPU Sets
        
        Args:
            pid: 进程ID
            cpu_set_ids: CPU Set ID列表
            
        Returns:
            是否设置成功
        """
        if not self._cpu_sets_supported:
            logger.warning("CPU Sets API不受支持")
            return False
        
        if not cpu_set_ids:
            logger.warning("CPU Set ID列表为空")
            return False
        
        handle = self._open_process(pid, PROCESS_SET_INFORMATION)
        if handle is None:
            return False
        
        try:
            # 创建CPU Set ID数组
            cpu_set_array = (wintypes.DWORD * len(cpu_set_ids))(*cpu_set_ids)
            
            # 调用API
            result = self.kernel32.SetProcessDefaultCpuSets(
                handle,
                cpu_set_array,
                len(cpu_set_ids)
            )
            
            if result == 0:
                error = ctypes.get_last_error()
                logger.error(f"SetProcessDefaultCpuSets 失败: 错误码 {error}")
                return False
            
            logger.debug(f"成功设置进程 {pid} 的默认CPU Sets: {cpu_set_ids}")
            return True
            
        except Exception as e:
            logger.error(f"设置进程默认CPU Sets失败: {e}")
            return False
        finally:
            self._close_handle(handle)
    
    def get_process_default_cpu_sets(self, pid: int) -> Optional[List[int]]:
        """
        获取进程默认CPU Sets
        
        Args:
            pid: 进程ID
            
        Returns:
            CPU Set ID列表，失败返回None
        """
        if not self._cpu_sets_supported:
            return None
        
        handle = self._open_process(pid, PROCESS_QUERY_INFORMATION)
        if handle is None:
            return None
        
        try:
            # 获取所需缓冲区大小
            buffer_size = wintypes.DWORD(0)
            result = self.kernel32.GetProcessDefaultCpuSets(
                handle, None, 0, ctypes.byref(buffer_size)
            )
            
            if result == 0 and ctypes.get_last_error() != ERROR_INSUFFICIENT_BUFFER:
                return None
            
            if buffer_size.value == 0:
                return []
            
            # 分配缓冲区
            buffer = (wintypes.DWORD * buffer_size.value)()
            
            # 获取CPU Sets
            result = self.kernel32.GetProcessDefaultCpuSets(
                handle, buffer, buffer_size, ctypes.byref(buffer_size)
            )
            
            if result == 0:
                return None
            
            return list(buffer[:buffer_size.value])
            
        except Exception as e:
            logger.error(f"获取进程默认CPU Sets失败: {e}")
            return None
        finally:
            self._close_handle(handle)
    
    def set_thread_selected_cpu_sets(self, tid: int, cpu_set_ids: List[int]) -> bool:
        """
        设置线程选定的CPU Sets
        
        Args:
            tid: 线程ID
            cpu_set_ids: CPU Set ID列表
            
        Returns:
            是否设置成功
        """
        if not self._cpu_sets_supported:
            logger.warning("CPU Sets API不受支持")
            return False
        
        if not cpu_set_ids:
            logger.warning("CPU Set ID列表为空")
            return False
        
        handle = self._open_thread(tid, THREAD_SET_LIMITED_INFORMATION)
        if handle is None:
            return False
        
        try:
            # 创建CPU Set ID数组
            cpu_set_array = (wintypes.DWORD * len(cpu_set_ids))(*cpu_set_ids)
            
            # 调用API
            result = self.kernel32.SetThreadSelectedCpuSets(
                handle,
                cpu_set_array,
                len(cpu_set_ids)
            )
            
            if result == 0:
                error = ctypes.get_last_error()
                logger.error(f"SetThreadSelectedCpuSets 失败: 错误码 {error}")
                return False
            
            logger.debug(f"成功设置线程 {tid} 的CPU Sets: {cpu_set_ids}")
            return True
            
        except Exception as e:
            logger.error(f"设置线程CPU Sets失败: {e}")
            return False
        finally:
            self._close_handle(handle)
    
    def get_thread_selected_cpu_sets(self, tid: int) -> Optional[List[int]]:
        """
        获取线程选定的CPU Sets
        
        Args:
            tid: 线程ID
            
        Returns:
            CPU Set ID列表，失败返回None
        """
        if not self._cpu_sets_supported:
            return None
        
        handle = self._open_thread(tid, THREAD_QUERY_INFORMATION)
        if handle is None:
            return None
        
        try:
            # 获取所需缓冲区大小
            buffer_size = wintypes.DWORD(0)
            result = self.kernel32.GetThreadSelectedCpuSets(
                handle, None, 0, ctypes.byref(buffer_size)
            )
            
            if result == 0 and ctypes.get_last_error() != ERROR_INSUFFICIENT_BUFFER:
                return None
            
            if buffer_size.value == 0:
                return []
            
            # 分配缓冲区
            buffer = (wintypes.DWORD * buffer_size.value)()
            
            # 获取CPU Sets
            result = self.kernel32.GetThreadSelectedCpuSets(
                handle, buffer, buffer_size, ctypes.byref(buffer_size)
            )
            
            if result == 0:
                return None
            
            return list(buffer[:buffer_size.value])
            
        except Exception as e:
            logger.error(f"获取线程CPU Sets失败: {e}")
            return None
        finally:
            self._close_handle(handle)
    
    # ========== Thread Affinity API ==========
    
    def set_thread_affinity_mask(self, tid: int, mask: int) -> Optional[int]:
        """
        设置线程亲和度掩码
        
        Args:
            tid: 线程ID
            mask: 亲和度位掩码
            
        Returns:
            之前的亲和度掩码，失败返回None
        """
        handle = self._open_thread(tid, THREAD_SET_INFORMATION)
        if handle is None:
            return None
        
        try:
            result = self.kernel32.SetThreadAffinityMask(handle, ULONG_PTR(mask))
            
            if result == 0:
                error = ctypes.get_last_error()
                logger.error(f"SetThreadAffinityMask 失败: 错误码 {error}")
                return None
            
            logger.debug(f"成功设置线程 {tid} 的亲和度掩码: {mask:#x}")
            return int(result)
            
        except Exception as e:
            logger.error(f"设置线程亲和度掩码失败: {e}")
            return None
        finally:
            self._close_handle(handle)
    
    def get_thread_affinity_mask(self, tid: int) -> Optional[int]:
        """
        获取线程亲和度掩码
        
        Args:
            tid: 线程ID
            
        Returns:
            亲和度掩码，失败返回None
        """
        handle = self._open_thread(tid, THREAD_QUERY_INFORMATION)
        if handle is None:
            return None
        
        try:
            # 使用SetThreadAffinityMask获取当前掩码
            # 传入0表示查询当前掩码而不改变
            result = self.kernel32.SetThreadAffinityMask(handle, ULONG_PTR(0))
            
            if result == 0:
                error = ctypes.get_last_error()
                logger.error(f"获取线程亲和度掩码失败: 错误码 {error}")
                return None
            
            # 恢复原掩码
            self.kernel32.SetThreadAffinityMask(handle, result)
            
            return int(result)
            
        except Exception as e:
            logger.error(f"获取线程亲和度掩码失败: {e}")
            return None
        finally:
            self._close_handle(handle)
    
    def set_thread_group_affinity(self, tid: int, group: int, mask: int) -> bool:
        """
        设置线程处理器组亲和度
        
        用于多处理器组系统（>64核心）
        
        Args:
            tid: 线程ID
            group: 处理器组
            mask: 亲和度位掩码
            
        Returns:
            是否设置成功
        """
        handle = self._open_thread(tid, THREAD_SET_INFORMATION)
        if handle is None:
            return False
        
        try:
            affinity = GROUP_AFFINITY()
            affinity.Mask = ULONG_PTR(mask)
            affinity.Group = group
            
            result = self.kernel32.SetThreadGroupAffinity(
                handle,
                ctypes.byref(affinity),
                None
            )
            
            if result == 0:
                error = ctypes.get_last_error()
                logger.error(f"SetThreadGroupAffinity 失败: 错误码 {error}")
                return False
            
            logger.debug(f"成功设置线程 {tid} 的组亲和度: Group={group}, Mask={mask:#x}")
            return True
            
        except Exception as e:
            logger.error(f"设置线程组亲和度失败: {e}")
            return False
        finally:
            self._close_handle(handle)
    
    def get_thread_group_affinity(self, tid: int) -> Optional[Tuple[int, int]]:
        """
        获取线程处理器组亲和度
        
        Args:
            tid: 线程ID
            
        Returns:
            (group, mask)元组，失败返回None
        """
        handle = self._open_thread(tid, THREAD_QUERY_INFORMATION)
        if handle is None:
            return None
        
        try:
            affinity = GROUP_AFFINITY()
            
            result = self.kernel32.GetThreadGroupAffinity(
                handle,
                ctypes.byref(affinity)
            )
            
            if result == 0:
                error = ctypes.get_last_error()
                logger.error(f"GetThreadGroupAffinity 失败: 错误码 {error}")
                return None
            
            return (affinity.Group, int(affinity.Mask))
            
        except Exception as e:
            logger.error(f"获取线程组亲和度失败: {e}")
            return None
        finally:
            self._close_handle(handle)
    
    def set_process_affinity_mask(self, pid: int, mask: int) -> bool:
        """
        设置进程亲和度掩码
        
        Args:
            pid: 进程ID
            mask: 亲和度位掩码
            
        Returns:
            是否设置成功
        """
        handle = self._open_process(pid, PROCESS_SET_INFORMATION)
        if handle is None:
            return False
        
        try:
            result = self.kernel32.SetProcessAffinityMask(handle, ULONG_PTR(mask))
            
            if result == 0:
                error = ctypes.get_last_error()
                logger.error(f"SetProcessAffinityMask 失败: 错误码 {error}")
                return False
            
            logger.debug(f"成功设置进程 {pid} 的亲和度掩码: {mask:#x}")
            return True
            
        except Exception as e:
            logger.error(f"设置进程亲和度掩码失败: {e}")
            return False
        finally:
            self._close_handle(handle)
    
    def get_process_affinity_mask(self, pid: int) -> Optional[Tuple[int, int]]:
        """
        获取进程亲和度掩码
        
        Args:
            pid: 进程ID
            
        Returns:
            (process_mask, system_mask)元组，失败返回None
        """
        handle = self._open_process(pid, PROCESS_QUERY_INFORMATION)
        if handle is None:
            return None
        
        try:
            process_mask = ULONG_PTR()
            system_mask = ULONG_PTR()
            
            result = self.kernel32.GetProcessAffinityMask(
                handle,
                ctypes.byref(process_mask),
                ctypes.byref(system_mask)
            )
            
            if result == 0:
                error = ctypes.get_last_error()
                logger.error(f"GetProcessAffinityMask 失败: 错误码 {error}")
                return None
            
            return (int(process_mask.value), int(system_mask.value))
            
        except Exception as e:
            logger.error(f"获取进程亲和度掩码失败: {e}")
            return None
        finally:
            self._close_handle(handle)
    
    def get_current_thread_id(self) -> int:
        """获取当前线程ID"""
        return self.kernel32.GetCurrentThreadId()
    
    def get_current_process_id(self) -> int:
        """获取当前进程ID"""
        return self.kernel32.GetCurrentProcessId()
    
    def get_system_info(self) -> Dict:
        """
        获取系统信息
        
        Returns:
            包含系统信息的字典
        """
        try:
            class SYSTEM_INFO(ctypes.Structure):
                _fields_ = [
                    ("wProcessorArchitecture", wintypes.WORD),
                    ("wReserved", wintypes.WORD),
                    ("dwPageSize", wintypes.DWORD),
                    ("lpMinimumApplicationAddress", wintypes.LPVOID),
                    ("lpMaximumApplicationAddress", wintypes.LPVOID),
                    ("dwActiveProcessorMask", ULONG_PTR),
                    ("dwNumberOfProcessors", wintypes.DWORD),
                    ("dwProcessorType", wintypes.DWORD),
                    ("dwAllocationGranularity", wintypes.DWORD),
                    ("wProcessorLevel", wintypes.WORD),
                    ("wProcessorRevision", wintypes.WORD),
                ]
            
            sys_info = SYSTEM_INFO()
            self.kernel32.GetSystemInfo(ctypes.byref(sys_info))
            
            return {
                'processor_architecture': sys_info.wProcessorArchitecture,
                'page_size': sys_info.dwPageSize,
                'active_processor_mask': int(sys_info.dwActiveProcessorMask),
                'number_of_processors': sys_info.dwNumberOfProcessors,
                'processor_type': sys_info.dwProcessorType,
                'processor_level': sys_info.wProcessorLevel,
                'processor_revision': sys_info.wProcessorRevision,
            }
        except Exception as e:
            logger.error(f"获取系统信息失败: {e}")
            return {}


# 创建全局API实例
_affinity_api = None

def get_affinity_api() -> WindowsAffinityAPI:
    """获取全局WindowsAffinityAPI实例"""
    global _affinity_api
    if _affinity_api is None:
        _affinity_api = WindowsAffinityAPI()
    return _affinity_api
