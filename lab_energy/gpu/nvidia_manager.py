"""
NVIDIA GPU管理模块

使用pynvml实现NVIDIA显卡的监控和功耗控制
"""

import logging
import subprocess
from typing import Optional, List, Dict, Any

from .gpu_types import (
    GPUInfo, GPUVendor, GPUPowerState, GPUOptimizationConfig,
    GPUInitializationError, GPUNotFoundError, GPUPowerLimitError
)

logger = logging.getLogger(__name__)


class NVIDIAManager:
    """NVIDIA GPU管理器
    
    使用pynvml库监控和控制NVIDIA显卡
    """
    
    def __init__(self):
        self._initialized = False
        self._device_count = 0
        self._handles: Dict[int, Any] = {}
        self._pynvml = None
        
    def initialize(self) -> bool:
        """初始化NVIDIA GPU监控
        
        Returns:
            是否初始化成功
        """
        try:
            import pynvml
            self._pynvml = pynvml
            
            # 初始化NVML
            pynvml.nvmlInit()
            self._initialized = True
            
            # 获取GPU数量
            self._device_count = pynvml.nvmlDeviceGetCount()
            logger.info(f"检测到 {self._device_count} 个NVIDIA GPU")
            
            # 缓存设备句柄
            for i in range(self._device_count):
                self._handles[i] = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            return True
            
        except ImportError:
            logger.warning("pynvml未安装，无法监控NVIDIA GPU")
            return False
        except Exception as e:
            logger.error(f"初始化NVIDIA GPU监控失败: {e}")
            return False
    
    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self._initialized
    
    def get_device_count(self) -> int:
        """获取GPU数量"""
        return self._device_count
    
    def _get_handle(self, gpu_id: int) -> Any:
        """获取GPU句柄"""
        if gpu_id not in self._handles:
            raise GPUNotFoundError(f"GPU {gpu_id} 未找到")
        return self._handles[gpu_id]
    
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
            handle = self._get_handle(gpu_id)
            pynvml = self._pynvml
            
            # 获取利用率
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            # 获取显存信息
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # 获取功耗信息 (mW -> W)
            try:
                power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            except pynvml.NVMLError:
                power_draw = 0.0
            
            try:
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
            except pynvml.NVMLError:
                power_limit = 0.0
            
            try:
                power_max = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1] / 1000.0
            except pynvml.NVMLError:
                power_max = power_limit
            
            # 获取温度
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except pynvml.NVMLError:
                temp = 0.0
            
            # 获取频率
            try:
                clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
            except pynvml.NVMLError:
                clock = 0
            
            try:
                max_clock = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
            except pynvml.NVMLError:
                max_clock = clock
            
            # 获取名称
            try:
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
            except pynvml.NVMLError:
                name = f"NVIDIA GPU {gpu_id}"
            
            return GPUInfo(
                gpu_id=gpu_id,
                vendor=GPUVendor.NVIDIA,
                name=name,
                gpu_util=util.gpu,
                memory_util=util.memory,
                memory_used_mb=mem.used // (1024 * 1024),
                memory_total_mb=mem.total // (1024 * 1024),
                power_draw_w=power_draw,
                power_limit_w=power_limit,
                power_max_w=power_max,
                temperature=temp,
                clock_mhz=clock,
                max_clock_mhz=max_clock
            )
            
        except Exception as e:
            logger.error(f"获取GPU {gpu_id} 信息失败: {e}")
            return None
    
    def get_all_gpus(self) -> List[GPUInfo]:
        """获取所有GPU信息"""
        gpus = []
        for i in range(self._device_count):
            info = self.get_gpu_info(i)
            if info:
                gpus.append(info)
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
            # 使用nvidia-smi设置功耗限制
            result = subprocess.run(
                ['nvidia-smi', '-i', str(gpu_id), '-pl', str(watts)],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                logger.info(f"GPU {gpu_id} 功耗限制设置为 {watts}W")
                return True
            else:
                logger.error(f"设置GPU {gpu_id} 功耗限制失败: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"设置GPU {gpu_id} 功耗限制失败: {e}")
            return False
    
    def set_clock_offset(self, gpu_id: int, offset_mhz: int) -> bool:
        """设置GPU时钟偏移
        
        Args:
            gpu_id: GPU编号
            offset_mhz: 时钟偏移（MHz）
            
        Returns:
            是否设置成功
        """
        if not self._initialized:
            return False
        
        try:
            # 使用nvidia-smi设置时钟偏移
            result = subprocess.run(
                ['nvidia-smi', '-i', str(gpu_id), '-lgc', f'{offset_mhz}'],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                logger.info(f"GPU {gpu_id} 时钟偏移设置为 {offset_mhz}MHz")
                return True
            else:
                logger.error(f"设置GPU {gpu_id} 时钟偏移失败: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"设置GPU {gpu_id} 时钟偏移失败: {e}")
            return False
    
    def reset_clock_offset(self, gpu_id: int) -> bool:
        """重置GPU时钟偏移
        
        Args:
            gpu_id: GPU编号
            
        Returns:
            是否重置成功
        """
        if not self._initialized:
            return False
        
        try:
            result = subprocess.run(
                ['nvidia-smi', '-i', str(gpu_id), '-rgc'],
                capture_output=True,
                text=True,
                check=False
            )
            
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"重置GPU {gpu_id} 时钟偏移失败: {e}")
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
        info = self.get_gpu_info(gpu_id)
        if not info:
            return GPUPowerState.IDLE
        
        util = info.gpu_util
        
        if util < 5.0:
            return GPUPowerState.IDLE
        elif util < 30.0:
            return GPUPowerState.LIGHT
        elif util < 70.0:
            return GPUPowerState.MODERATE
        else:
            return GPUPowerState.HEAVY
    
    def apply_power_optimization(
        self, 
        gpu_id: int = 0, 
        config: Optional[GPUOptimizationConfig] = None
    ) -> bool:
        """应用功耗优化
        
        根据当前状态自动调整功耗限制
        
        Args:
            gpu_id: GPU编号
            config: 优化配置
            
        Returns:
            是否应用成功
        """
        if not self._initialized:
            return False
        
        if config is None:
            config = GPUOptimizationConfig()
        
        try:
            info = self.get_gpu_info(gpu_id)
            if not info:
                return False
            
            state = self.detect_power_state(gpu_id)
            
            # 根据状态计算目标功耗
            if state == GPUPowerState.IDLE:
                target_percent = config.idle_power_percent
            elif state == GPUPowerState.LIGHT:
                target_percent = config.light_power_percent
            elif state == GPUPowerState.MODERATE:
                target_percent = config.moderate_power_percent
            else:  # HEAVY
                target_percent = config.heavy_power_percent
            
            # 计算目标功耗
            target_watts = info.power_max_w * (target_percent / 100.0)
            
            # 应用功耗限制
            success = self.set_power_limit(gpu_id, target_watts)
            
            if success:
                logger.info(
                    f"GPU {gpu_id} 功耗优化: {state.value} -> {target_watts:.1f}W "
                    f"({target_percent}% of max)"
                )
            
            return success
            
        except Exception as e:
            logger.error(f"应用GPU {gpu_id} 功耗优化失败: {e}")
            return False
    
    def shutdown(self):
        """关闭NVIDIA管理器，释放资源"""
        if self._initialized and self._pynvml:
            try:
                self._pynvml.nvmlShutdown()
                logger.info("NVIDIA GPU管理器已关闭")
            except Exception as e:
                logger.error(f"关闭NVIDIA GPU管理器失败: {e}")
            finally:
                self._initialized = False
                self._handles.clear()
