"""
激进Boost控制器

实现1ms内将CPU频率提升到最高的激进Boost策略，同时保持平时低功耗运行。

特性:
- 1ms响应的性能模式切换
- Windows Game Mode API集成
- 电源节流控制
- Core Parking控制
- 自动负载检测和Boost

示例:
    >>> from lab_energy.cpu import BoostController, PowerMode
    >>> boost = BoostController()
    >>> 
    >>> # 进入激进Boost模式
    >>> boost.enter_aggressive_mode()
    >>> 
    >>> # 执行高性能任务...
    >>> 
    >>> # 恢复正常
    >>> boost.exit_aggressive_mode()
"""

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Any
import logging

from .power_api import WindowsPowerAPI


# 配置日志
logger = logging.getLogger(__name__)


class PowerMode(Enum):
    """电源模式"""
    POWERSAVE = "powersave"       # 节能模式
    BALANCED = "balanced"         # 平衡模式
    PERFORMANCE = "performance"   # 高性能模式
    AGGRESSIVE = "aggressive"     # 激进Boost模式（1ms响应）


@dataclass
class BoostProfile:
    """Boost配置档案"""
    mode: PowerMode
    min_cpu_percent: int          # 最小处理器状态
    max_cpu_percent: int          # 最大处理器状态
    disable_parking: bool         # 禁用Core Parking
    disable_throttling: bool      # 禁用电源节流
    boost_duration_ms: int        # Boost持续时间
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "mode": self.mode.value,
            "min_cpu_percent": self.min_cpu_percent,
            "max_cpu_percent": self.max_cpu_percent,
            "disable_parking": self.disable_parking,
            "disable_throttling": self.disable_throttling,
            "boost_duration_ms": self.boost_duration_ms,
        }


# 预定义Boost配置
BOOST_PROFILES: Dict[PowerMode, BoostProfile] = {
    PowerMode.POWERSAVE: BoostProfile(
        mode=PowerMode.POWERSAVE,
        min_cpu_percent=5,
        max_cpu_percent=50,
        disable_parking=False,
        disable_throttling=False,
        boost_duration_ms=0
    ),
    PowerMode.BALANCED: BoostProfile(
        mode=PowerMode.BALANCED,
        min_cpu_percent=5,
        max_cpu_percent=100,
        disable_parking=False,
        disable_throttling=False,
        boost_duration_ms=0
    ),
    PowerMode.PERFORMANCE: BoostProfile(
        mode=PowerMode.PERFORMANCE,
        min_cpu_percent=50,
        max_cpu_percent=100,
        disable_parking=True,
        disable_throttling=True,
        boost_duration_ms=0
    ),
    PowerMode.AGGRESSIVE: BoostProfile(
        mode=PowerMode.AGGRESSIVE,
        min_cpu_percent=100,
        max_cpu_percent=100,
        disable_parking=True,
        disable_throttling=True,
        boost_duration_ms=5000
    ),
}


class GameModeAPI:
    """
    Windows Game Mode API封装
    
    使用Windows 10 Game Mode API优化游戏/高性能应用性能。
    """
    
    def __init__(self):
        self._available = self._check_availability()
        
    def _check_availability(self) -> bool:
        """检查Game Mode API是否可用"""
        try:
            import ctypes
            from ctypes import wintypes
            
            # 尝试加载相关API
            kernel32 = ctypes.windll.kernel32
            # HasUserPreferenceForGameMode 是内部API，可能不存在
            # 我们通过注册表检查Game Mode是否启用
            
            import winreg
            try:
                key = winreg.OpenKey(
                    winreg.HKEY_CURRENT_USER,
                    r"Software\Microsoft\GameBar"
                )
                value, _ = winreg.QueryValueEx(key, "AllowAutoGameMode")
                winreg.CloseKey(key)
                return value == 1
            except (WindowsError, FileNotFoundError):
                return False
                
        except Exception:
            return False
    
    @property
    def available(self) -> bool:
        """Game Mode API是否可用"""
        return self._available
    
    def enable_game_mode(self) -> bool:
        """
        启用Game Mode
        
        Returns:
            是否成功启用
        """
        try:
            import winreg
            
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\GameBar",
                0,
                winreg.KEY_WRITE
            )
            winreg.SetValueEx(key, "AllowAutoGameMode", 0, winreg.REG_DWORD, 1)
            winreg.SetValueEx(key, "AutoGameModeEnabled", 0, winreg.REG_DWORD, 1)
            winreg.CloseKey(key)
            
            logger.info("Game Mode已启用")
            return True
            
        except Exception as e:
            logger.warning(f"启用Game Mode失败: {e}")
            return False
    
    def disable_game_mode(self) -> bool:
        """
        禁用Game Mode
        
        Returns:
            是否成功禁用
        """
        try:
            import winreg
            
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\GameBar",
                0,
                winreg.KEY_WRITE
            )
            winreg.SetValueEx(key, "AutoGameModeEnabled", 0, winreg.REG_DWORD, 0)
            winreg.CloseKey(key)
            
            logger.info("Game Mode已禁用")
            return True
            
        except Exception as e:
            logger.warning(f"禁用Game Mode失败: {e}")
            return False


class CPUMonitor:
    """CPU负载监控器"""
    
    def __init__(self, sample_interval: float = 0.1):
        """
        初始化CPU监控器
        
        Args:
            sample_interval: 采样间隔（秒）
        """
        self.sample_interval = sample_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[float], None]] = []
        self._current_load = 0.0
        
    def start(self):
        """启动监控"""
        if self._running:
            return
            
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.debug("CPU监控已启动")
        
    def stop(self):
        """停止监控"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        logger.debug("CPU监控已停止")
        
    def _monitor_loop(self):
        """监控循环"""
        try:
            import psutil
            
            while self._running:
                # 获取CPU使用率
                self._current_load = psutil.cpu_percent(interval=self.sample_interval)
                
                # 触发回调
                for callback in self._callbacks:
                    try:
                        callback(self._current_load)
                    except Exception as e:
                        logger.error(f"CPU监控回调异常: {e}")
                        
        except ImportError:
            logger.warning("psutil未安装，CPU监控不可用")
            self._running = False
        except Exception as e:
            logger.error(f"CPU监控循环异常: {e}")
            self._running = False
    
    def register_callback(self, callback: Callable[[float], None]):
        """
        注册CPU负载回调
        
        Args:
            callback: 回调函数，接收CPU负载百分比
        """
        self._callbacks.append(callback)
        
    def unregister_callback(self, callback: Callable[[float], None]):
        """注销回调"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def get_current_load(self) -> float:
        """获取当前CPU负载"""
        return self._current_load


class BoostController:
    """
    激进Boost控制器
    
    实现1ms内将CPU频率提升到最高的策略
    使用Windows Game Mode API和电源管理API
    """
    
    # Windows电源方案GUID
    POWER_PLANS: Dict[PowerMode, str] = {
        PowerMode.POWERSAVE: "a1841308-3541-4fab-bc81-f71556f20b4a",
        PowerMode.BALANCED: "381b4222-f694-41f0-9685-ff5bb260df2e",
        PowerMode.PERFORMANCE: "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c",
        PowerMode.AGGRESSIVE: "e9a42b02-d5df-448d-aa00-03f14749eb61",  # 卓越性能
    }
    
    def __init__(self):
        self._current_mode: PowerMode = PowerMode.BALANCED
        self._boost_start_time: Optional[float] = None
        self._boost_active: bool = False
        self._boost_timer: Optional[threading.Timer] = None
        self._power_api = WindowsPowerAPI()
        self._game_mode = GameModeAPI()
        self._cpu_monitor = CPUMonitor()
        self._auto_boost_enabled = False
        self._load_threshold = 80.0
        self._auto_boost_callback: Optional[Callable] = None
        self._lock = threading.Lock()
        
        # 保存原始状态用于恢复
        self._original_power_plan: Optional[str] = None
        self._original_min_state: Optional[int] = None
        self._original_max_state: Optional[int] = None
        
    @property
    def current_mode(self) -> PowerMode:
        """当前电源模式"""
        return self._current_mode
    
    @property
    def boost_active(self) -> bool:
        """Boost是否处于活动状态"""
        return self._boost_active
    
    @property
    def boost_elapsed_ms(self) -> float:
        """Boost已持续时间（毫秒）"""
        if self._boost_start_time is None:
            return 0.0
        return (time.perf_counter() - self._boost_start_time) * 1000
    
    def boost_now(self, duration_ms: int = 5000) -> bool:
        """
        立即执行激进Boost（1ms响应目标）
        
        执行步骤:
        1. 禁用当前进程的电源节流
        2. 设置最小处理器状态为100%
        3. 切换到高性能电源方案
        4. 禁用Core Parking
        
        Args:
            duration_ms: Boost持续时间（毫秒）
            
        Returns:
            是否成功启动Boost
        """
        with self._lock:
            start_time = time.perf_counter()
            
            try:
                # 保存当前状态（如果尚未保存）
                if not self._boost_active:
                    self._save_current_state()
                
                # 1. 禁用当前进程电源节流
                if not self._power_api.set_process_power_throttling(0, disable_throttling=True):
                    logger.warning(f"禁用电源节流失败: {self._power_api.last_error}")
                
                # 2. 设置处理器状态为100%
                if not self._power_api.set_processor_state(100, 100):
                    logger.warning(f"设置处理器状态失败: {self._power_api.last_error}")
                
                # 3. 切换到激进电源方案
                plan_guid = self.POWER_PLANS[PowerMode.AGGRESSIVE]
                if not self._power_api.set_power_plan(plan_guid):
                    # 如果卓越性能不可用，使用高性能
                    plan_guid = self.POWER_PLANS[PowerMode.PERFORMANCE]
                    if not self._power_api.set_power_plan(plan_guid):
                        logger.warning(f"设置电源方案失败: {self._power_api.last_error}")
                
                # 4. 禁用Core Parking
                if not self._power_api.disable_core_parking(True):
                    logger.warning(f"禁用Core Parking失败: {self._power_api.last_error}")
                
                # 5. 启用Game Mode
                if self._game_mode.available:
                    self._game_mode.enable_game_mode()
                
                # 更新状态
                self._boost_active = True
                self._boost_start_time = time.perf_counter()
                self._current_mode = PowerMode.AGGRESSIVE
                
                # 设置自动恢复定时器
                if duration_ms > 0:
                    self._schedule_boost_exit(duration_ms)
                
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                logger.info(f"激进Boost已启动，耗时: {elapsed_ms:.2f}ms")
                
                return True
                
            except Exception as e:
                logger.error(f"启动Boost失败: {e}")
                return False
    
    def set_power_throttling(self, pid: int, disable: bool) -> bool:
        """
        设置进程电源节流状态
        
        使用SetProcessInformation + ProcessPowerThrottling
        
        Args:
            pid: 进程ID，0表示当前进程
            disable: True=禁用节流（全速）, False=允许节流
            
        Returns:
            是否设置成功
        """
        result = self._power_api.set_process_power_throttling(pid, disable)
        if result:
            logger.info(f"进程 {pid} 电源节流已{'禁用' if disable else '启用'}")
        else:
            logger.warning(f"设置进程 {pid} 电源节流失败: {self._power_api.last_error}")
        return result
    
    def set_cpu_throttling(self, min_percent: int, max_percent: int) -> bool:
        """
        设置CPU节流状态
        
        使用powercfg设置处理器状态
        
        Args:
            min_percent: 最小处理器状态 (0-100)
            max_percent: 最大处理器状态 (0-100)
            
        Returns:
            是否设置成功
        """
        result = self._power_api.set_processor_state(min_percent, max_percent)
        if result:
            logger.info(f"CPU节流已设置: 最小={min_percent}%, 最大={max_percent}%")
        else:
            logger.warning(f"设置CPU节流失败: {self._power_api.last_error}")
        return result
    
    def set_power_plan(self, mode: PowerMode) -> bool:
        """
        切换Windows电源方案
        
        Args:
            mode: 电源模式
            
        Returns:
            是否切换成功
        """
        plan_guid = self.POWER_PLANS.get(mode)
        if not plan_guid:
            logger.error(f"未知的电源模式: {mode}")
            return False
            
        result = self._power_api.set_power_plan(plan_guid)
        if result:
            self._current_mode = mode
            logger.info(f"电源方案已切换至: {mode.value}")
        else:
            logger.warning(f"切换电源方案失败: {self._power_api.last_error}")
        return result
    
    def disable_core_parking(self, disable: bool = True) -> bool:
        """
        禁用/启用Core Parking
        
        Args:
            disable: True=禁用Core Parking, False=允许
            
        Returns:
            是否设置成功
        """
        result = self._power_api.disable_core_parking(disable)
        if result:
            logger.info(f"Core Parking已{'禁用' if disable else '启用'}")
        else:
            logger.warning(f"设置Core Parking失败: {self._power_api.last_error}")
        return result
    
    def enter_aggressive_mode(self) -> bool:
        """
        进入激进Boost模式
        
        组合调用:
        - set_power_throttling(disable=True)
        - set_cpu_throttling(100, 100)
        - set_power_plan(PowerMode.AGGRESSIVE)
        - disable_core_parking(True)
        
        Returns:
            是否成功进入
        """
        return self.boost_now(duration_ms=0)  # 0表示不自动恢复
    
    def exit_aggressive_mode(self) -> bool:
        """
        退出激进Boost模式，恢复正常
        
        Returns:
            是否成功退出
        """
        with self._lock:
            try:
                # 取消定时器
                if self._boost_timer:
                    self._boost_timer.cancel()
                    self._boost_timer = None
                
                # 恢复原始状态
                if self._original_power_plan:
                    self._power_api.set_power_plan(self._original_power_plan)
                
                if self._original_min_state is not None and self._original_max_state is not None:
                    self._power_api.set_processor_state(
                        self._original_min_state,
                        self._original_max_state
                    )
                
                # 重新启用电源节流（恢复默认）
                self._power_api.set_process_power_throttling(0, disable_throttling=False)
                
                # 禁用Game Mode
                if self._game_mode.available:
                    self._game_mode.disable_game_mode()
                
                # 更新状态
                self._boost_active = False
                self._boost_start_time = None
                self._current_mode = PowerMode.BALANCED
                
                # 清除保存的状态
                self._original_power_plan = None
                self._original_min_state = None
                self._original_max_state = None
                
                logger.info("已退出激进Boost模式")
                return True
                
            except Exception as e:
                logger.error(f"退出激进Boost模式失败: {e}")
                return False
    
    def auto_boost_on_load(
        self, 
        load_threshold: float = 80.0,
        callback: Optional[Callable] = None
    ):
        """
        根据CPU负载自动Boost
        
        Args:
            load_threshold: 触发Boost的负载阈值(%)
            callback: Boost触发时的回调函数
        """
        self._load_threshold = load_threshold
        self._auto_boost_callback = callback
        self._auto_boost_enabled = True
        
        # 注册负载监控回调
        self._cpu_monitor.register_callback(self._on_cpu_load)
        
        # 启动监控
        if not self._cpu_monitor._running:
            self._cpu_monitor.start()
        
        logger.info(f"自动Boost已启用，阈值: {load_threshold}%")
    
    def stop_auto_boost(self):
        """停止自动Boost"""
        self._auto_boost_enabled = False
        self._cpu_monitor.unregister_callback(self._on_cpu_load)
        logger.info("自动Boost已停止")
    
    def _on_cpu_load(self, load: float):
        """
        CPU负载回调
        
        Args:
            load: CPU负载百分比
        """
        if not self._auto_boost_enabled:
            return
            
        if load >= self._load_threshold and not self._boost_active:
            logger.info(f"CPU负载 {load:.1f}% 超过阈值 {self._load_threshold}%，触发自动Boost")
            
            # 触发Boost
            self.boost_now(duration_ms=3000)
            
            # 执行回调
            if self._auto_boost_callback:
                try:
                    self._auto_boost_callback()
                except Exception as e:
                    logger.error(f"自动Boost回调异常: {e}")
    
    def _save_current_state(self):
        """保存当前电源状态"""
        try:
            self._original_power_plan = self._power_api.get_current_power_plan()
            self._original_min_state, self._original_max_state = \
                self._power_api.get_processor_state()
            
            logger.debug(
                f"保存原始状态: 电源方案={self._original_power_plan}, "
                f"处理器状态={self._original_min_state}%-{self._original_max_state}%"
            )
        except Exception as e:
            logger.warning(f"保存当前状态失败: {e}")
    
    def _schedule_boost_exit(self, duration_ms: int):
        """
        安排Boost自动退出
        
        Args:
            duration_ms: 持续时间（毫秒）
        """
        # 取消之前的定时器
        if self._boost_timer:
            self._boost_timer.cancel()
        
        # 创建新定时器
        self._boost_timer = threading.Timer(
            duration_ms / 1000.0,
            self.exit_aggressive_mode
        )
        self._boost_timer.daemon = True
        self._boost_timer.start()
        
        logger.debug(f"已安排 {duration_ms}ms 后自动退出Boost")
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取控制器状态
        
        Returns:
            状态字典
        """
        return {
            "current_mode": self._current_mode.value,
            "boost_active": self._boost_active,
            "boost_elapsed_ms": self.boost_elapsed_ms,
            "auto_boost_enabled": self._auto_boost_enabled,
            "load_threshold": self._load_threshold,
            "current_cpu_load": self._cpu_monitor.get_current_load(),
            "game_mode_available": self._game_mode.available,
            "admin_privileges": self._power_api.is_admin(),
        }
    
    def apply_profile(self, profile: BoostProfile) -> bool:
        """
        应用Boost配置档案
        
        Args:
            profile: Boost配置档案
            
        Returns:
            是否应用成功
        """
        try:
            # 设置CPU节流
            if not self.set_cpu_throttling(profile.min_cpu_percent, profile.max_cpu_percent):
                return False
            
            # 设置Core Parking
            if not self.disable_core_parking(profile.disable_parking):
                return False
            
            # 设置电源节流
            if not self.set_power_throttling(0, profile.disable_throttling):
                return False
            
            # 设置电源方案
            if not self.set_power_plan(profile.mode):
                return False
            
            self._current_mode = profile.mode
            
            # 如果配置了持续时间，设置自动恢复
            if profile.boost_duration_ms > 0:
                self._schedule_boost_exit(profile.boost_duration_ms)
            
            logger.info(f"已应用配置档案: {profile.mode.value}")
            return True
            
        except Exception as e:
            logger.error(f"应用配置档案失败: {e}")
            return False
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口，确保恢复状态"""
        if self._boost_active:
            self.exit_aggressive_mode()
        self.stop_auto_boost()
        self._cpu_monitor.stop()


# 便捷函数
def quick_boost(duration_ms: int = 5000) -> bool:
    """
    快速Boost（便捷函数）
    
    Args:
        duration_ms: Boost持续时间（毫秒）
        
    Returns:
        是否成功
    """
    controller = BoostController()
    return controller.boost_now(duration_ms)


def set_performance_mode() -> bool:
    """
    设置高性能模式（便捷函数）
    
    Returns:
        是否成功
    """
    controller = BoostController()
    return controller.set_power_plan(PowerMode.PERFORMANCE)


def set_powersave_mode() -> bool:
    """
    设置节能模式（便捷函数）
    
    Returns:
        是否成功
    """
    controller = BoostController()
    return controller.set_power_plan(PowerMode.POWERSAVE)


if __name__ == "__main__":
    # 简单测试
    import sys
    
    print("激进Boost控制器测试")
    print("=" * 50)
    
    controller = BoostController()
    status = controller.get_status()
    
    print(f"管理员权限: {status['admin_privileges']}")
    print(f"Game Mode可用: {status['game_mode_available']}")
    print(f"当前模式: {status['current_mode']}")
    print()
    
    # 测试Boost
    print("测试激进Boost（3秒）...")
    if controller.boost_now(duration_ms=3000):
        print("Boost已启动")
        time.sleep(1)
        print(f"Boost持续时间: {controller.boost_elapsed_ms:.2f}ms")
        time.sleep(3)
        print(f"Boost状态: {controller.boost_active}")
    else:
        print("Boost启动失败")
        print(f"错误: {controller._power_api.last_error}")
