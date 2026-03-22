"""
AMD GPU管理模块

使用amdsmi或替代方案实现AMD显卡的监控和功耗控制
"""

import logging
import subprocess
import re
from typing import Optional, List, Dict, Any

from .gpu_types import (
    GPUInfo, GPUVendor, GPUPowerState, GPUOptimizationConfig,
    GPUNotFoundError
)

logger = logging.getLogger(__name__)


class AMDManager:
    """AMD GPU管理器
    
    尝试使用amdsmi库监控和控制AMD显卡
    如果amdsmi不可用，则使用rocm-smi作为备选方案
    """
    
    def __init__(self):
        self._initialized = False
        self._device_count = 0
        self._handles: Dict[int, Any] = {}
        self._amdsmi = None
        self._use_rocm_smi = False
        
    def initialize(self) -> bool:
        """初始化AMD GPU监控
        
        Returns:
            是否初始化成功
        """
        # 首先尝试使用amdsmi
        try:
            import amdsmi
            self._amdsmi = amdsmi
            
            # 初始化amdsmi
            amdsmi.amdsmi_init()
            self._initialized = True
            
            # 获取GPU数量
            devices = amdsmi.amdsmi_get_device_handles()
            self._device_count = len(devices)
            
            # 缓存设备句柄
            for i, device in enumerate(devices):
                self._handles[i] = device
            
            logger.info(f"检测到 {self._device_count} 个AMD GPU (使用amdsmi)")
            return True
            
        except ImportError:
            logger.debug("amdsmi未安装，尝试使用rocm-smi")
        except Exception as e:
            logger.debug(f"初始化amdsmi失败: {e}")
        
        # 备选方案：使用rocm-smi
        if self._check_rocm_smi():
            self._use_rocm_smi = True
            self._initialized = True
            logger.info(f"检测到 {self._device_count} 个AMD GPU (使用rocm-smi)")
            return True
        
        logger.warning("无法初始化AMD GPU监控")
        return False
    
    def _check_rocm_smi(self) -> bool:
        """检查rocm-smi是否可用并获取GPU数量"""
        try:
            result = subprocess.run(
                ['rocm-smi', '--showid'],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                # 解析GPU数量
                lines = result.stdout.strip().split('\n')
                count = 0
                for line in lines:
                    if line.startswith('GPU'):
                        count += 1
                
                self._device_count = count
                return count > 0
            
            return False
            
        except FileNotFoundError:
            return False
        except Exception as e:
            logger.debug(f"检查rocm-smi失败: {e}")
            return False
    
    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self._initialized
    
    def get_device_count(self) -> int:
        """获取GPU数量"""
        return self._device_count
    
    def get_gpu_info(self, gpu_id: int = 0) -> Optional[GPUInfo]:
        """获取GPU信息
        
        Args:
            gpu_id: GPU编号
            
        Returns:
            GPU信息
        """
        if not self._initialized:
            return None
        
        if self._amdsmi:
            return self._get_gpu_info_amdsmi(gpu_id)
        elif self._use_rocm_smi:
            return self._get_gpu_info_rocm_smi(gpu_id)
        
        return None
    
    def _get_gpu_info_amdsmi(self, gpu_id: int) -> Optional[GPUInfo]:
        """使用amdsmi获取GPU信息"""
        try:
            if gpu_id not in self._handles:
                raise GPUNotFoundError(f"GPU {gpu_id} 未找到")
            
            device = self._handles[gpu_id]
            amdsmi = self._amdsmi
            
            # 获取GPU名称
            try:
                name = amdsmi.amdsmi_get_gpu_vendor_name(device)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
            except:
                name = f"AMD GPU {gpu_id}"
            
            # 获取利用率
            try:
                util_info = amdsmi.amdsmi_get_gpu_activity(device)
                gpu_util = util_info.get('gfx_activity', 0)
            except:
                gpu_util = 0.0
            
            # 获取显存信息
            try:
                mem_info = amdsmi.amdsmi_get_gpu_memory_total(device)
                mem_total = mem_info.get('vram', 0) // (1024 * 1024)  # 转换为MB
            except:
                mem_total = 0
            
            try:
                mem_used_info = amdsmi.amdsmi_get_gpu_memory_usage(device)
                mem_used = mem_used_info.get('vram', 0) // (1024 * 1024)
            except:
                mem_used = 0
            
            # 计算显存利用率
            memory_util = (mem_used / mem_total * 100) if mem_total > 0 else 0.0
            
            # 获取功耗
            try:
                power = amdsmi.amdsmi_get_power_info(device)
                power_draw = power.get('average_power', 0) / 1000.0  # mW -> W
                power_limit = power.get('power_limit', 0) / 1000.0
            except:
                power_draw = 0.0
                power_limit = 0.0
            
            # 获取温度
            try:
                temp = amdsmi.amdsmi_get_temp_metric(
                    device, amdsmi.AMDSMI_TEMPERATURE_TYPE_EDGE
                )
                temperature = temp.get('temperature', 0)
            except:
                temperature = 0.0
            
            # 获取频率
            try:
                clock = amdsmi.amdsmi_get_clock_info(device, amdsmi.AMDSMI_CLK_TYPE_GFX)
                clock_mhz = clock.get('clk', 0)
                max_clock_mhz = clock.get('max_clk', clock_mhz)
            except:
                clock_mhz = 0
                max_clock_mhz = 0
            
            return GPUInfo(
                gpu_id=gpu_id,
                vendor=GPUVendor.AMD,
                name=name,
                gpu_util=gpu_util,
                memory_util=memory_util,
                memory_used_mb=mem_used,
                memory_total_mb=mem_total,
                power_draw_w=power_draw,
                power_limit_w=power_limit,
                power_max_w=power_limit,  # AMD通常只有一个功耗限制
                temperature=temperature,
                clock_mhz=clock_mhz,
                max_clock_mhz=max_clock_mhz
            )
            
        except Exception as e:
            logger.error(f"使用amdsmi获取GPU {gpu_id} 信息失败: {e}")
            return None
    
    def _get_gpu_info_rocm_smi(self, gpu_id: int) -> Optional[GPUInfo]:
        """使用rocm-smi获取GPU信息"""
        try:
            # 获取所有信息
            result = subprocess.run(
                ['rocm-smi', '--showid', '--showuse', '--showmeminfo', 'vram',
                 '--showpower', '--showtemp', '--showclk', 'freq'],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                return None
            
            lines = result.stdout.strip().split('\n')
            
            # 解析GPU信息
            gpu_data = {}
            for line in lines:
                if line.startswith(f'GPU[{gpu_id}]'):
                    parts = line.split()
                    # 这里需要根据rocm-smi的输出格式进行解析
                    # 这是一个简化的实现
                    gpu_data['id'] = gpu_id
                    gpu_data['name'] = f"AMD GPU {gpu_id}"
            
            if not gpu_data:
                return None
            
            # 由于rocm-smi输出格式复杂，这里返回一个基本的GPUInfo
            # 实际使用时需要根据具体输出格式完善解析逻辑
            return GPUInfo(
                gpu_id=gpu_id,
                vendor=GPUVendor.AMD,
                name=gpu_data.get('name', f"AMD GPU {gpu_id}"),
                gpu_util=0.0,
                memory_util=0.0,
                memory_used_mb=0,
                memory_total_mb=0,
                power_draw_w=0.0,
                power_limit_w=0.0,
                power_max_w=0.0,
                temperature=0.0,
                clock_mhz=0,
                max_clock_mhz=0
            )
            
        except Exception as e:
            logger.error(f"使用rocm-smi获取GPU {gpu_id} 信息失败: {e}")
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
            if self._amdsmi:
                # 使用amdsmi设置功耗限制
                if gpu_id not in self._handles:
                    return False
                device = self._handles[gpu_id]
                # 转换为毫瓦
                self._amdsmi.amdsmi_set_power_cap(device, int(watts * 1000))
                logger.info(f"AMD GPU {gpu_id} 功耗限制设置为 {watts}W")
                return True
            
            elif self._use_rocm_smi:
                # 使用rocm-smi设置功耗限制
                result = subprocess.run(
                    ['rocm-smi', '--setpoweroverdrive', str(int(watts)),
                     '-d', str(gpu_id)],
                    capture_output=True,
                    text=True,
                    check=False
                )
                
                if result.returncode == 0:
                    logger.info(f"AMD GPU {gpu_id} 功耗限制设置为 {watts}W")
                    return True
                else:
                    logger.error(f"设置AMD GPU {gpu_id} 功耗限制失败: {result.stderr}")
                    return False
            
            return False
            
        except Exception as e:
            logger.error(f"设置AMD GPU {gpu_id} 功耗限制失败: {e}")
            return False
    
    def set_clock_level(self, gpu_id: int, level: str) -> bool:
        """设置GPU时钟级别
        
        Args:
            gpu_id: GPU编号
            level: 时钟级别 ('low', 'high', 'auto')
            
        Returns:
            是否设置成功
        """
        if not self._initialized or not self._use_rocm_smi:
            return False
        
        try:
            if level == 'low':
                # 设置低功耗模式
                result = subprocess.run(
                    ['rocm-smi', '--setperflevel', 'low', '-d', str(gpu_id)],
                    capture_output=True,
                    text=True,
                    check=False
                )
            elif level == 'high':
                # 设置高性能模式
                result = subprocess.run(
                    ['rocm-smi', '--setperflevel', 'high', '-d', str(gpu_id)],
                    capture_output=True,
                    text=True,
                    check=False
                )
            else:
                # 自动模式
                result = subprocess.run(
                    ['rocm-smi', '--setperflevel', 'auto', '-d', str(gpu_id)],
                    capture_output=True,
                    text=True,
                    check=False
                )
            
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"设置AMD GPU {gpu_id} 时钟级别失败: {e}")
            return False
    
    def detect_power_state(self, gpu_id: int = 0) -> GPUPowerState:
        """检测GPU电源状态
        
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
            state = self.detect_power_state(gpu_id)
            
            # AMD GPU使用性能级别而非直接设置功耗限制
            if state == GPUPowerState.IDLE:
                success = self.set_clock_level(gpu_id, 'low')
                logger.info(f"AMD GPU {gpu_id} 设置为低功耗模式 ({state.value})")
            elif state == GPUPowerState.HEAVY:
                success = self.set_clock_level(gpu_id, 'high')
                logger.info(f"AMD GPU {gpu_id} 设置为高性能模式 ({state.value})")
            else:
                success = self.set_clock_level(gpu_id, 'auto')
                logger.info(f"AMD GPU {gpu_id} 设置为自动模式 ({state.value})")
            
            return success
            
        except Exception as e:
            logger.error(f"应用AMD GPU {gpu_id} 功耗优化失败: {e}")
            return False
    
    def shutdown(self):
        """关闭AMD管理器，释放资源"""
        if self._initialized and self._amdsmi:
            try:
                self._amdsmi.amdsmi_shut_down()
                logger.info("AMD GPU管理器已关闭")
            except Exception as e:
                logger.error(f"关闭AMD GPU管理器失败: {e}")
            finally:
                self._initialized = False
                self._handles.clear()
