"""
GPU节能管理器主模块

提供统一的GPU监控和功耗控制接口，支持NVIDIA和AMD显卡
"""

import logging
import threading
import time
from typing import Optional, List, Callable
from dataclasses import dataclass

from .gpu_types import (
    GPUInfo, GPUVendor, GPUPowerState, GPUOptimizationConfig,
    GPUInitializationError, GPUNotFoundError
)
from .nvidia_manager import NVIDIAManager
from .amd_manager import AMDManager

logger = logging.getLogger(__name__)


@dataclass
class GPUStats:
    """GPU统计信息"""
    total_gpus: int = 0
    nvidia_gpus: int = 0
    amd_gpus: int = 0
    total_power_draw: float = 0.0
    avg_temperature: float = 0.0
    avg_utilization: float = 0.0


class GPUManager:
    """GPU节能管理器
    
    支持NVIDIA和AMD显卡的监控和功耗控制
    提供统一的接口管理所有GPU设备
    """
    
    def __init__(self):
        self._nvidia_manager: Optional[NVIDIAManager] = None
        self._amd_manager: Optional[AMDManager] = None
        self._initialized = False
        self._auto_optimize_thread: Optional[threading.Thread] = None
        self._stop_auto_optimize = threading.Event()
        self._optimization_config = GPUOptimizationConfig()
        self._callbacks: List[Callable[[int, GPUPowerState], None]] = []
        
    def initialize(self) -> bool:
        """初始化GPU管理器
        
        自动检测并初始化所有可用的GPU管理器
        
        Returns:
            是否初始化成功（至少一个GPU管理器成功初始化）
        """
        success = False
        
        # 尝试初始化NVIDIA管理器
        try:
            self._nvidia_manager = NVIDIAManager()
            if self._nvidia_manager.initialize():
                logger.info(f"NVIDIA GPU管理器初始化成功，检测到 {self._nvidia_manager.get_device_count()} 个GPU")
                success = True
            else:
                self._nvidia_manager = None
        except Exception as e:
            logger.debug(f"初始化NVIDIA管理器失败: {e}")
            self._nvidia_manager = None
        
        # 尝试初始化AMD管理器
        try:
            self._amd_manager = AMDManager()
            if self._amd_manager.initialize():
                logger.info(f"AMD GPU管理器初始化成功，检测到 {self._amd_manager.get_device_count()} 个GPU")
                success = True
            else:
                self._amd_manager = None
        except Exception as e:
            logger.debug(f"初始化AMD管理器失败: {e}")
            self._amd_manager = None
        
        self._initialized = success
        
        if not success:
            logger.warning("未检测到支持的GPU设备")
        else:
            logger.info(f"GPU管理器初始化完成，共 {self.get_gpu_count()} 个GPU")
        
        return success
    
    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self._initialized
    
    def get_gpu_count(self) -> int:
        """获取GPU总数"""
        count = 0
        if self._nvidia_manager:
            count += self._nvidia_manager.get_device_count()
        if self._amd_manager:
            count += self._amd_manager.get_device_count()
        return count
    
    def get_stats(self) -> GPUStats:
        """获取GPU统计信息"""
        stats = GPUStats()
        
        all_gpus = self.get_all_gpus()
        stats.total_gpus = len(all_gpus)
        
        if not all_gpus:
            return stats
        
        total_power = 0.0
        total_temp = 0.0
        total_util = 0.0
        
        for gpu in all_gpus:
            total_power += gpu.power_draw_w
            total_temp += gpu.temperature
            total_util += gpu.gpu_util
            
            if gpu.vendor == GPUVendor.NVIDIA:
                stats.nvidia_gpus += 1
            elif gpu.vendor == GPUVendor.AMD:
                stats.amd_gpus += 1
        
        stats.total_power_draw = total_power
        stats.avg_temperature = total_temp / len(all_gpus)
        stats.avg_utilization = total_util / len(all_gpus)
        
        return stats
    
    def _get_manager_for_gpu(self, gpu_id: int) -> tuple:
        """获取指定GPU对应的管理器和本地ID
        
        Args:
            gpu_id: 全局GPU编号
            
        Returns:
            (管理器, 本地GPU ID) 元组
        """
        # NVIDIA GPUs优先
        if self._nvidia_manager:
            nvidia_count = self._nvidia_manager.get_device_count()
            if gpu_id < nvidia_count:
                return self._nvidia_manager, gpu_id
            gpu_id -= nvidia_count
        
        # AMD GPUs
        if self._amd_manager:
            if gpu_id < self._amd_manager.get_device_count():
                return self._amd_manager, gpu_id
        
        raise GPUNotFoundError(f"GPU {gpu_id} 未找到")
    
    def get_gpu_info(self, gpu_id: int = 0) -> Optional[GPUInfo]:
        """获取GPU信息
        
        Args:
            gpu_id: GPU编号
            
        Returns:
            GPU信息
        """
        if not self._initialized:
            return None
        
        try:
            manager, local_id = self._get_manager_for_gpu(gpu_id)
            info = manager.get_gpu_info(local_id)
            if info:
                # 更新全局GPU ID
                info.gpu_id = gpu_id
            return info
        except Exception as e:
            logger.error(f"获取GPU {gpu_id} 信息失败: {e}")
            return None
    
    def get_all_gpus(self) -> List[GPUInfo]:
        """获取所有GPU信息
        
        Returns:
            所有GPU信息列表
        """
        gpus = []
        global_id = 0
        
        # 获取NVIDIA GPUs
        if self._nvidia_manager:
            for i in range(self._nvidia_manager.get_device_count()):
                info = self._nvidia_manager.get_gpu_info(i)
                if info:
                    info.gpu_id = global_id
                    gpus.append(info)
                global_id += 1
        
        # 获取AMD GPUs
        if self._amd_manager:
            for i in range(self._amd_manager.get_device_count()):
                info = self._amd_manager.get_gpu_info(i)
                if info:
                    info.gpu_id = global_id
                    gpus.append(info)
                global_id += 1
        
        return gpus
    
    def set_power_limit(self, gpu_id: int, watts: float) -> bool:
        """设置GPU功耗限制
        
        Args:
            gpu_id: GPU编号
            watts: 功耗限制（瓦特）
            
        Returns:
            是否设置成功
        """
        if not self._initialized:
            return False
        
        try:
            manager, local_id = self._get_manager_for_gpu(gpu_id)
            return manager.set_power_limit(local_id, watts)
        except Exception as e:
            logger.error(f"设置GPU {gpu_id} 功耗限制失败: {e}")
            return False
    
    def set_clock_offset(self, gpu_id: int, offset_mhz: int) -> bool:
        """设置GPU时钟偏移（仅NVIDIA支持）
        
        Args:
            gpu_id: GPU编号
            offset_mhz: 时钟偏移（MHz）
            
        Returns:
            是否设置成功
        """
        if not self._initialized:
            return False
        
        try:
            manager, local_id = self._get_manager_for_gpu(gpu_id)
            if isinstance(manager, NVIDIAManager):
                return manager.set_clock_offset(local_id, offset_mhz)
            else:
                logger.warning(f"GPU {gpu_id} 不支持时钟偏移设置")
                return False
        except Exception as e:
            logger.error(f"设置GPU {gpu_id} 时钟偏移失败: {e}")
            return False
    
    def detect_power_state(self, gpu_id: int = 0) -> GPUPowerState:
        """检测GPU电源状态
        
        基于利用率判断:
        - IDLE: < 5%
        - LIGHT: 5-30%
        - MODERATE: 30-70%
        - HEAVY: > 70%
        
        Args:
            gpu_id: GPU编号
            
        Returns:
            电源状态
        """
        if not self._initialized:
            return GPUPowerState.IDLE
        
        try:
            manager, local_id = self._get_manager_for_gpu(gpu_id)
            return manager.detect_power_state(local_id)
        except Exception as e:
            logger.error(f"检测GPU {gpu_id} 电源状态失败: {e}")
            return GPUPowerState.IDLE
    
    def apply_power_optimization(self, gpu_id: int = 0) -> bool:
        """应用功耗优化
        
        根据当前状态自动调整功耗限制
        
        Args:
            gpu_id: GPU编号
            
        Returns:
            是否应用成功
        """
        if not self._initialized:
            return False
        
        try:
            manager, local_id = self._get_manager_for_gpu(gpu_id)
            
            # 获取优化前的状态
            old_state = manager.detect_power_state(local_id)
            
            # 应用优化
            success = manager.apply_power_optimization(local_id, self._optimization_config)
            
            if success:
                # 获取优化后的状态
                new_state = manager.detect_power_state(local_id)
                
                # 触发回调
                for callback in self._callbacks:
                    try:
                        callback(gpu_id, new_state)
                    except Exception as e:
                        logger.error(f"执行GPU状态回调失败: {e}")
            
            return success
            
        except Exception as e:
            logger.error(f"应用GPU {gpu_id} 功耗优化失败: {e}")
            return False
    
    def apply_all_optimizations(self) -> dict:
        """对所有GPU应用功耗优化
        
        Returns:
            各GPU优化结果字典 {gpu_id: success}
        """
        results = {}
        for gpu_id in range(self.get_gpu_count()):
            results[gpu_id] = self.apply_power_optimization(gpu_id)
        return results
    
    def auto_optimize(
        self, 
        interval_seconds: int = 60,
        config: Optional[GPUOptimizationConfig] = None
    ):
        """自动优化模式
        
        定期检测GPU状态并自动调整功耗
        
        Args:
            interval_seconds: 检测间隔（秒）
            config: 优化配置
        """
        if config:
            self._optimization_config = config
        
        self._stop_auto_optimize.clear()
        
        def optimize_loop():
            logger.info(f"启动GPU自动优化，间隔 {interval_seconds} 秒")
            
            while not self._stop_auto_optimize.is_set():
                try:
                    self.apply_all_optimizations()
                except Exception as e:
                    logger.error(f"自动优化失败: {e}")
                
                # 等待下一次优化或停止信号
                self._stop_auto_optimize.wait(interval_seconds)
            
            logger.info("GPU自动优化已停止")
        
        self._auto_optimize_thread = threading.Thread(
            target=optimize_loop,
            name="GPUAutoOptimize",
            daemon=True
        )
        self._auto_optimize_thread.start()
    
    def stop_auto_optimize(self):
        """停止自动优化"""
        if self._auto_optimize_thread and self._auto_optimize_thread.is_alive():
            self._stop_auto_optimize.set()
            self._auto_optimize_thread.join(timeout=5.0)
            logger.info("自动优化线程已停止")
    
    def register_state_callback(self, callback: Callable[[int, GPUPowerState], None]):
        """注册GPU状态变化回调
        
        Args:
            callback: 回调函数，参数为 (gpu_id, power_state)
        """
        self._callbacks.append(callback)
    
    def unregister_state_callback(self, callback: Callable[[int, GPUPowerState], None]):
        """注销GPU状态变化回调
        
        Args:
            callback: 要注销的回调函数
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def set_optimization_config(self, config: GPUOptimizationConfig):
        """设置优化配置
        
        Args:
            config: 优化配置
        """
        self._optimization_config = config
        logger.info("GPU优化配置已更新")
    
    def get_optimization_config(self) -> GPUOptimizationConfig:
        """获取当前优化配置"""
        return self._optimization_config
    
    def reset_power_limits(self) -> dict:
        """重置所有GPU功耗限制到最大值
        
        Returns:
            各GPU重置结果字典 {gpu_id: success}
        """
        results = {}
        
        for gpu_id in range(self.get_gpu_count()):
            try:
                info = self.get_gpu_info(gpu_id)
                if info:
                    results[gpu_id] = self.set_power_limit(gpu_id, info.power_max_w)
                else:
                    results[gpu_id] = False
            except Exception as e:
                logger.error(f"重置GPU {gpu_id} 功耗限制失败: {e}")
                results[gpu_id] = False
        
        return results
    
    def shutdown(self):
        """关闭GPU管理器，释放资源"""
        logger.info("正在关闭GPU管理器...")
        
        # 停止自动优化
        self.stop_auto_optimize()
        
        # 重置功耗限制
        self.reset_power_limits()
        
        # 关闭各管理器
        if self._nvidia_manager:
            self._nvidia_manager.shutdown()
            self._nvidia_manager = None
        
        if self._amd_manager:
            self._amd_manager.shutdown()
            self._amd_manager = None
        
        self._initialized = False
        self._callbacks.clear()
        
        logger.info("GPU管理器已关闭")
