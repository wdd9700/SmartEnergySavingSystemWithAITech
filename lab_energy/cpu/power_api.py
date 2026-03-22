"""
Windows电源管理API封装

提供对Windows电源管理API的Python封装，包括：
- 进程电源节流控制 (ProcessPowerThrottling)
- 电源方案切换
- 处理器状态设置
- Core Parking控制

要求:
    - Windows 10 2004+ (支持ProcessPowerThrottling)
    - 管理员权限（修改电源方案）

示例:
    >>> from lab_energy.cpu.power_api import WindowsPowerAPI
    >>> # 禁用当前进程的电源节流
    >>> WindowsPowerAPI.set_process_power_throttling(0, disable_throttling=True)
    >>> # 切换到高性能电源方案
    >>> WindowsPowerAPI.set_power_plan("8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c")
"""

import ctypes
import subprocess
import sys
from ctypes import wintypes
from typing import Optional, Tuple


class PROCESS_POWER_THROTTLING_STATE(ctypes.Structure):
    """
    进程电源节流状态结构
    
    typedef struct _PROCESS_POWER_THROTTLING_STATE {
        ULONG Version;
        ULONG ControlMask;
        ULONG StateMask;
    } PROCESS_POWER_THROTTLING_STATE;
    """
    _fields_ = [
        ("Version", wintypes.ULONG),
        ("ControlMask", wintypes.ULONG),
        ("StateMask", wintypes.ULONG),
    ]


class WindowsPowerAPI:
    """Windows电源管理API封装"""
    
    # ProcessPowerThrottling常量
    PROCESS_POWER_THROTTLING_STATE_VERSION = 1
    PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1
    PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION = 0x2
    
    # Windows API常量
    PROCESS_SET_INFORMATION = 0x0200
    
    # 电源方案GUID
    POWER_PLAN_POWERSAVE = "a1841308-3541-4fab-bc81-f71556f20b4a"
    POWER_PLAN_BALANCED = "381b4222-f694-41f0-9685-ff5bb260df2e"
    POWER_PLAN_PERFORMANCE = "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"
    POWER_PLAN_AGGRESSIVE = "e9a42b02-d5df-448d-aa00-03f14749eb61"  # 卓越性能
    
    # 电源设置GUID
    SUB_PROCESSOR = "54533251-82be-4824-96c1-47b60b740d00"
    PROC_THROTTLE_MIN = "893dee8e-2bef-41e0-89c6-b55d0929964c"
    PROC_THROTTLE_MAX = "bc5038f7-23e0-4960-96da-33abaf5935ec"
    
    def __init__(self):
        self._kernel32 = ctypes.windll.kernel32
        self._powrprof = ctypes.windll.powrprof
        self._last_error: Optional[str] = None
        
    @property
    def last_error(self) -> Optional[str]:
        """获取最后一次错误信息"""
        return self._last_error
    
    def set_process_power_throttling(self, pid: int, disable_throttling: bool) -> bool:
        """
        设置进程电源节流
        
        使用SetProcessInformation + ProcessPowerThrottling控制进程电源节流。
        需要Windows 10 2004+版本支持。
        
        Args:
            pid: 进程ID，0表示当前进程
            disable_throttling: True=禁用节流（全速运行），False=允许节流
            
        Returns:
            是否设置成功
            
        Example:
            >>> api = WindowsPowerAPI()
            >>> api.set_process_power_throttling(0, disable_throttling=True)
            True
        """
        try:
            # ProcessPowerThrottling = 4 (Windows SDK)
            ProcessPowerThrottling = 4
            
            # 打开进程获取句柄
            if pid == 0:
                process_handle = self._kernel32.GetCurrentProcess()
            else:
                process_handle = self._kernel32.OpenProcess(
                    self.PROCESS_SET_INFORMATION,
                    False,
                    pid
                )
                if not process_handle:
                    self._last_error = f"无法打开进程 {pid}"
                    return False
            
            try:
                # 设置电源节流状态
                control_mask = self.PROCESS_POWER_THROTTLING_EXECUTION_SPEED
                # StateMask = 0 表示禁用节流，= ControlMask 表示启用节流
                state_mask = 0 if disable_throttling else control_mask
                
                power_throttling = PROCESS_POWER_THROTTLING_STATE()
                power_throttling.Version = self.PROCESS_POWER_THROTTLING_STATE_VERSION
                power_throttling.ControlMask = control_mask
                power_throttling.StateMask = state_mask
                
                # 调用SetProcessInformation
                result = self._kernel32.SetProcessInformation(
                    process_handle,
                    ProcessPowerThrottling,
                    ctypes.byref(power_throttling),
                    ctypes.sizeof(power_throttling)
                )
                
                if result == 0:
                    error_code = ctypes.get_last_error()
                    self._last_error = f"SetProcessInformation失败，错误码: {error_code}"
                    return False
                    
                return True
                
            finally:
                if pid != 0 and process_handle:
                    self._kernel32.CloseHandle(process_handle)
                    
        except Exception as e:
            self._last_error = f"设置进程电源节流异常: {str(e)}"
            return False
    
    def set_thread_power_throttling(self, thread_id: int, disable_throttling: bool) -> bool:
        """
        设置线程电源节流（如果支持）
        
        注意: Windows 10 2004+ 主要支持进程级节流，线程级节流支持有限。
        
        Args:
            thread_id: 线程ID，0表示当前线程
            disable_throttling: True=禁用节流，False=允许节流
            
        Returns:
            是否设置成功
        """
        try:
            # ThreadPowerThrottling = 6 (Windows SDK)
            ThreadPowerThrottling = 6
            
            # 获取线程句柄
            if thread_id == 0:
                thread_handle = self._kernel32.GetCurrentThread()
            else:
                THREAD_SET_INFORMATION = 0x0020
                thread_handle = self._kernel32.OpenThread(
                    THREAD_SET_INFORMATION,
                    False,
                    thread_id
                )
                if not thread_handle:
                    self._last_error = f"无法打开线程 {thread_id}"
                    return False
            
            try:
                # 设置电源节流状态
                control_mask = self.PROCESS_POWER_THROTTLING_EXECUTION_SPEED
                state_mask = 0 if disable_throttling else control_mask
                
                power_throttling = PROCESS_POWER_THROTTLING_STATE()
                power_throttling.Version = self.PROCESS_POWER_THROTTLING_STATE_VERSION
                power_throttling.ControlMask = control_mask
                power_throttling.StateMask = state_mask
                
                result = self._kernel32.SetThreadInformation(
                    thread_handle,
                    ThreadPowerThrottling,
                    ctypes.byref(power_throttling),
                    ctypes.sizeof(power_throttling)
                )
                
                if result == 0:
                    error_code = ctypes.get_last_error()
                    self._last_error = f"SetThreadInformation失败，错误码: {error_code}"
                    return False
                    
                return True
                
            finally:
                if thread_id != 0 and thread_handle:
                    self._kernel32.CloseHandle(thread_handle)
                    
        except Exception as e:
            self._last_error = f"设置线程电源节流异常: {str(e)}"
            return False
    
    def set_power_plan(self, plan_guid: str) -> bool:
        """
        切换Windows电源方案
        
        使用powercfg命令或PowerSetActiveScheme API设置电源方案。
        需要管理员权限。
        
        Args:
            plan_guid: 电源方案GUID字符串
            
        Returns:
            是否切换成功
            
        Example:
            >>> api = WindowsPowerAPI()
            >>> api.set_power_plan(WindowsPowerAPI.POWER_PLAN_PERFORMANCE)
            True
        """
        try:
            # 方法1: 使用powercfg命令
            result = subprocess.run(
                ["powercfg", "/setactive", plan_guid],
                capture_output=True,
                text=True,
                shell=False
            )
            
            if result.returncode == 0:
                return True
            
            # 方法2: 使用PowerSetActiveScheme API（备用）
            return self._set_power_plan_api(plan_guid)
            
        except Exception as e:
            self._last_error = f"设置电源方案异常: {str(e)}"
            return False
    
    def _set_power_plan_api(self, plan_guid: str) -> bool:
        """
        使用PowerSetActiveScheme API设置电源方案
        
        Args:
            plan_guid: 电源方案GUID字符串
            
        Returns:
            是否设置成功
        """
        try:
            # 转换GUID字符串为GUID结构
            import uuid
            guid = uuid.UUID(plan_guid)
            
            # 创建GUID结构
            class GUID(ctypes.Structure):
                _fields_ = [
                    ("Data1", wintypes.ULONG),
                    ("Data2", wintypes.USHORT),
                    ("Data3", wintypes.USHORT),
                    ("Data4", wintypes.BYTE * 8),
                ]
            
            power_guid = GUID()
            power_guid.Data1 = guid.fields[0]
            power_guid.Data2 = guid.fields[1]
            power_guid.Data3 = guid.fields[2]
            for i in range(8):
                power_guid.Data4[i] = guid.fields[3][i]
            
            # 调用PowerSetActiveScheme
            result = self._powrprof.PowerSetActiveScheme(
                None,  # Root power key (None = 当前用户)
                ctypes.byref(power_guid)
            )
            
            if result != 0:  # ERROR_SUCCESS = 0
                self._last_error = f"PowerSetActiveScheme失败，错误码: {result}"
                return False
                
            return True
            
        except Exception as e:
            self._last_error = f"API设置电源方案异常: {str(e)}"
            return False
    
    def set_processor_state(
        self, 
        min_percent: int, 
        max_percent: int, 
        ac_power: bool = True
    ) -> bool:
        """
        设置处理器状态（最小/最大处理器状态）
        
        使用powercfg命令设置处理器节流状态。
        需要管理员权限。
        
        Args:
            min_percent: 最小处理器状态 (0-100)
            max_percent: 最大处理器状态 (0-100)
            ac_power: True=交流电源设置, False=直流电源设置
            
        Returns:
            是否设置成功
            
        Example:
            >>> api = WindowsPowerAPI()
            >>> api.set_processor_state(100, 100)  # 禁用节流，全速运行
            True
        """
        try:
            # 验证参数
            min_percent = max(0, min(100, min_percent))
            max_percent = max(0, min(100, max_percent))
            
            # 构建powercfg参数
            power_type = "/SETACVALUEINDEX" if ac_power else "/SETDCVALUEINDEX"
            scheme = "SCHEME_CURRENT"
            
            # 设置最小处理器状态
            result_min = subprocess.run(
                [
                    "powercfg", power_type, scheme, 
                    self.SUB_PROCESSOR, self.PROC_THROTTLE_MIN,
                    str(min_percent)
                ],
                capture_output=True,
                text=True,
                shell=False
            )
            
            # 设置最大处理器状态
            result_max = subprocess.run(
                [
                    "powercfg", power_type, scheme,
                    self.SUB_PROCESSOR, self.PROC_THROTTLE_MAX,
                    str(max_percent)
                ],
                capture_output=True,
                text=True,
                shell=False
            )
            
            # 应用设置
            result_apply = subprocess.run(
                ["powercfg", "/SETACTIVE", "SCHEME_CURRENT"],
                capture_output=True,
                text=True,
                shell=False
            )
            
            if result_min.returncode != 0 or result_max.returncode != 0:
                self._last_error = f"设置处理器状态失败: {result_min.stderr} {result_max.stderr}"
                return False
                
            return True
            
        except Exception as e:
            self._last_error = f"设置处理器状态异常: {str(e)}"
            return False
    
    def disable_core_parking(self, disable: bool = True) -> bool:
        """
        禁用/启用Core Parking
        
        通过设置最小处理器状态为100%来禁用Core Parking。
        需要管理员权限。
        
        Args:
            disable: True=禁用Core Parking, False=允许Core Parking
            
        Returns:
            是否设置成功
        """
        try:
            if disable:
                # 禁用Core Parking: 设置最小处理器状态为100%
                return self.set_processor_state(100, 100)
            else:
                # 启用Core Parking: 恢复默认设置（通常5%）
                return self.set_processor_state(5, 100)
                
        except Exception as e:
            self._last_error = f"设置Core Parking异常: {str(e)}"
            return False
    
    def get_current_power_plan(self) -> Optional[str]:
        """
        获取当前电源方案GUID
        
        Returns:
            当前电源方案GUID，失败返回None
        """
        try:
            result = subprocess.run(
                ["powercfg", "/getactivescheme"],
                capture_output=True,
                text=True,
                shell=False
            )
            
            if result.returncode == 0:
                # 解析输出: "电源方案 GUID: 381b4222-f694-41f0-9685-ff5bb260df2e  (平衡)"
                output = result.stdout.strip()
                if "GUID:" in output:
                    parts = output.split("GUID:")[1].strip().split()
                    return parts[0]
                    
            return None
            
        except Exception as e:
            self._last_error = f"获取当前电源方案异常: {str(e)}"
            return None
    
    def get_processor_state(self) -> Tuple[Optional[int], Optional[int]]:
        """
        获取当前处理器状态
        
        Returns:
            (最小处理器状态, 最大处理器状态)，失败返回(None, None)
        """
        try:
            result = subprocess.run(
                ["powercfg", "/query", "SCHEME_CURRENT", self.SUB_PROCESSOR, self.PROC_THROTTLE_MIN],
                capture_output=True,
                text=True,
                shell=False
            )
            
            min_state = None
            max_state = None
            
            if result.returncode == 0:
                # 解析输出查找当前交流电源设置
                for line in result.stdout.split('\n'):
                    if "当前交流电源设置索引" in line or "Current AC Power Setting Index" in line:
                        try:
                            # 格式: 0x00000064 (100)
                            value_str = line.split('(')[1].split(')')[0]
                            min_state = int(value_str)
                        except (IndexError, ValueError):
                            pass
            
            result = subprocess.run(
                ["powercfg", "/query", "SCHEME_CURRENT", self.SUB_PROCESSOR, self.PROC_THROTTLE_MAX],
                capture_output=True,
                text=True,
                shell=False
            )
            
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if "当前交流电源设置索引" in line or "Current AC Power Setting Index" in line:
                        try:
                            value_str = line.split('(')[1].split(')')[0]
                            max_state = int(value_str)
                        except (IndexError, ValueError):
                            pass
            
            return min_state, max_state
            
        except Exception as e:
            self._last_error = f"获取处理器状态异常: {str(e)}"
            return None, None
    
    def is_admin(self) -> bool:
        """
        检查是否以管理员权限运行
        
        Returns:
            是否拥有管理员权限
        """
        try:
            return ctypes.windll.shell32.IsUserAnAdmin()
        except Exception:
            return False


# 便捷函数
def set_process_high_performance(pid: int = 0) -> bool:
    """
    将进程设置为高性能模式（便捷函数）
    
    Args:
        pid: 进程ID，0表示当前进程
        
    Returns:
        是否设置成功
    """
    api = WindowsPowerAPI()
    return api.set_process_power_throttling(pid, disable_throttling=True)


def boost_system_performance() -> bool:
    """
    提升系统整体性能（便捷函数）
    
    禁用Core Parking并设置处理器状态为高性能。
    需要管理员权限。
    
    Returns:
        是否设置成功
    """
    api = WindowsPowerAPI()
    return api.disable_core_parking(True)


if __name__ == "__main__":
    # 简单测试
    api = WindowsPowerAPI()
    
    print("Windows电源API测试")
    print("-" * 40)
    
    # 检查管理员权限
    print(f"管理员权限: {api.is_admin()}")
    
    # 获取当前电源方案
    current_plan = api.get_current_power_plan()
    print(f"当前电源方案: {current_plan}")
    
    # 获取处理器状态
    min_state, max_state = api.get_processor_state()
    print(f"处理器状态: 最小={min_state}%, 最大={max_state}%")
    
    # 测试禁用当前进程节流
    result = api.set_process_power_throttling(0, disable_throttling=True)
    print(f"禁用当前进程节流: {'成功' if result else '失败'}")
    if not result:
        print(f"  错误: {api.last_error}")
