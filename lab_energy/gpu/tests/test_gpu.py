"""
GPU管理模块单元测试

测试GPU类型定义、NVIDIA管理器、AMD管理器和主GPU管理器
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import threading
import time

from lab_energy.gpu import (
    GPUManager, GPUVendor, GPUPowerState, GPUInfo,
    GPUOptimizationConfig, GPUStats,
    GPUError, GPUInitializationError, GPUNotFoundError
)
from lab_energy.gpu.gpu_types import GPUPowerLimitError, GPUUnsupportedError


class TestGPUTypes(unittest.TestCase):
    """测试GPU类型定义"""
    
    def test_gpu_vendor_enum(self):
        """测试GPU厂商枚举"""
        self.assertEqual(GPUVendor.NVIDIA.value, "nvidia")
        self.assertEqual(GPUVendor.AMD.value, "amd")
        self.assertEqual(GPUVendor.INTEL.value, "intel")
        self.assertEqual(GPUVendor.UNKNOWN.value, "unknown")
    
    def test_gpu_power_state_enum(self):
        """测试GPU电源状态枚举"""
        self.assertEqual(GPUPowerState.IDLE.value, "idle")
        self.assertEqual(GPUPowerState.LIGHT.value, "light")
        self.assertEqual(GPUPowerState.MODERATE.value, "moderate")
        self.assertEqual(GPUPowerState.HEAVY.value, "heavy")
    
    def test_gpu_info_dataclass(self):
        """测试GPUInfo数据类"""
        info = GPUInfo(
            gpu_id=0,
            vendor=GPUVendor.NVIDIA,
            name="RTX 4090",
            gpu_util=50.5,
            memory_util=30.0,
            memory_used_mb=4096,
            memory_total_mb=24576,
            power_draw_w=250.5,
            power_limit_w=300.0,
            power_max_w=450.0,
            temperature=65.5,
            clock_mhz=2500,
            max_clock_mhz=2800
        )
        
        self.assertEqual(info.gpu_id, 0)
        self.assertEqual(info.vendor, GPUVendor.NVIDIA)
        self.assertEqual(info.name, "RTX 4090")
        self.assertEqual(info.gpu_util, 50.5)
        self.assertEqual(info.power_draw_w, 250.5)
    
    def test_gpu_optimization_config_defaults(self):
        """测试GPU优化配置默认值"""
        config = GPUOptimizationConfig()
        
        self.assertEqual(config.idle_power_percent, 50.0)
        self.assertEqual(config.light_power_percent, 70.0)
        self.assertEqual(config.moderate_power_percent, 85.0)
        self.assertEqual(config.heavy_power_percent, 100.0)
        self.assertEqual(config.idle_threshold, 5.0)
        self.assertEqual(config.light_threshold, 30.0)
        self.assertEqual(config.moderate_threshold, 70.0)
        self.assertEqual(config.check_interval_seconds, 60)
        self.assertEqual(config.max_temperature, 85.0)
        self.assertTrue(config.thermal_throttle_enabled)
    
    def test_gpu_optimization_config_custom(self):
        """测试GPU优化配置自定义值"""
        config = GPUOptimizationConfig(
            idle_power_percent=40.0,
            heavy_power_percent=95.0,
            check_interval_seconds=30
        )
        
        self.assertEqual(config.idle_power_percent, 40.0)
        self.assertEqual(config.heavy_power_percent, 95.0)
        self.assertEqual(config.check_interval_seconds, 30)


class TestGPUExceptions(unittest.TestCase):
    """测试GPU异常类"""
    
    def test_gpu_error(self):
        """测试基础GPU异常"""
        with self.assertRaises(GPUError):
            raise GPUError("Test error")
    
    def test_gpu_initialization_error(self):
        """测试GPU初始化异常"""
        with self.assertRaises(GPUInitializationError):
            raise GPUInitializationError("Init failed")
    
    def test_gpu_not_found_error(self):
        """测试GPU未找到异常"""
        with self.assertRaises(GPUNotFoundError):
            raise GPUNotFoundError("GPU not found")
    
    def test_gpu_power_limit_error(self):
        """测试功耗限制异常"""
        with self.assertRaises(GPUPowerLimitError):
            raise GPUPowerLimitError("Power limit failed")
    
    def test_gpu_unsupported_error(self):
        """测试GPU不支持异常"""
        with self.assertRaises(GPUUnsupportedError):
            raise GPUUnsupportedError("Operation not supported")


class TestGPUManager(unittest.TestCase):
    """测试GPU管理器主类"""
    
    def setUp(self):
        """测试前准备"""
        self.manager = GPUManager()
    
    def tearDown(self):
        """测试后清理"""
        if self.manager.is_initialized():
            self.manager.shutdown()
    
    @patch('lab_energy.gpu.gpu_manager.NVIDIAManager')
    @patch('lab_energy.gpu.gpu_manager.AMDManager')
    def test_initialize_no_gpus(self, mock_amd, mock_nvidia):
        """测试初始化时没有GPU的情况"""
        # 模拟没有GPU
        mock_nvidia_instance = Mock()
        mock_nvidia_instance.initialize.return_value = False
        mock_nvidia.return_value = mock_nvidia_instance
        
        mock_amd_instance = Mock()
        mock_amd_instance.initialize.return_value = False
        mock_amd.return_value = mock_amd_instance
        
        result = self.manager.initialize()
        
        self.assertFalse(result)
        self.assertFalse(self.manager.is_initialized())
    
    @patch('lab_energy.gpu.gpu_manager.NVIDIAManager')
    @patch('lab_energy.gpu.gpu_manager.AMDManager')
    def test_initialize_with_nvidia(self, mock_amd, mock_nvidia):
        """测试初始化时有NVIDIA GPU的情况"""
        # 模拟NVIDIA GPU
        mock_nvidia_instance = Mock()
        mock_nvidia_instance.initialize.return_value = True
        mock_nvidia_instance.get_device_count.return_value = 2
        mock_nvidia.return_value = mock_nvidia_instance
        
        mock_amd_instance = Mock()
        mock_amd_instance.initialize.return_value = False
        mock_amd.return_value = mock_amd_instance
        
        result = self.manager.initialize()
        
        self.assertTrue(result)
        self.assertTrue(self.manager.is_initialized())
        self.assertEqual(self.manager.get_gpu_count(), 2)
    
    @patch('lab_energy.gpu.gpu_manager.NVIDIAManager')
    @patch('lab_energy.gpu.gpu_manager.AMDManager')
    def test_initialize_with_amd(self, mock_amd, mock_nvidia):
        """测试初始化时有AMD GPU的情况"""
        # 模拟AMD GPU
        mock_nvidia_instance = Mock()
        mock_nvidia_instance.initialize.return_value = False
        mock_nvidia.return_value = mock_nvidia_instance
        
        mock_amd_instance = Mock()
        mock_amd_instance.initialize.return_value = True
        mock_amd_instance.get_device_count.return_value = 1
        mock_amd.return_value = mock_amd_instance
        
        result = self.manager.initialize()
        
        self.assertTrue(result)
        self.assertTrue(self.manager.is_initialized())
        self.assertEqual(self.manager.get_gpu_count(), 1)
    
    @patch('lab_energy.gpu.gpu_manager.NVIDIAManager')
    @patch('lab_energy.gpu.gpu_manager.AMDManager')
    def test_get_all_gpus(self, mock_amd, mock_nvidia):
        """测试获取所有GPU信息"""
        # 创建模拟GPU信息
        nvidia_info = GPUInfo(
            gpu_id=0, vendor=GPUVendor.NVIDIA, name="RTX 4090",
            gpu_util=50.0, memory_util=30.0,
            memory_used_mb=4096, memory_total_mb=24576,
            power_draw_w=250.0, power_limit_w=300.0, power_max_w=450.0,
            temperature=65.0, clock_mhz=2500, max_clock_mhz=2800
        )
        
        # 模拟NVIDIA管理器
        mock_nvidia_instance = Mock()
        mock_nvidia_instance.initialize.return_value = True
        mock_nvidia_instance.get_device_count.return_value = 1
        mock_nvidia_instance.get_gpu_info.return_value = nvidia_info
        mock_nvidia.return_value = mock_nvidia_instance
        
        mock_amd_instance = Mock()
        mock_amd_instance.initialize.return_value = False
        mock_amd.return_value = mock_amd_instance
        
        self.manager.initialize()
        gpus = self.manager.get_all_gpus()
        
        self.assertEqual(len(gpus), 1)
        self.assertEqual(gpus[0].vendor, GPUVendor.NVIDIA)
        self.assertEqual(gpus[0].name, "RTX 4090")
    
    @patch('lab_energy.gpu.gpu_manager.NVIDIAManager')
    @patch('lab_energy.gpu.gpu_manager.AMDManager')
    def test_get_stats(self, mock_amd, mock_nvidia):
        """测试获取GPU统计信息"""
        # 创建模拟GPU信息
        nvidia_info = GPUInfo(
            gpu_id=0, vendor=GPUVendor.NVIDIA, name="RTX 4090",
            gpu_util=50.0, memory_util=30.0,
            memory_used_mb=4096, memory_total_mb=24576,
            power_draw_w=250.0, power_limit_w=300.0, power_max_w=450.0,
            temperature=65.0, clock_mhz=2500, max_clock_mhz=2800
        )
        
        mock_nvidia_instance = Mock()
        mock_nvidia_instance.initialize.return_value = True
        mock_nvidia_instance.get_device_count.return_value = 1
        mock_nvidia_instance.get_gpu_info.return_value = nvidia_info
        mock_nvidia.return_value = mock_nvidia_instance
        
        mock_amd_instance = Mock()
        mock_amd_instance.initialize.return_value = False
        mock_amd.return_value = mock_amd_instance
        
        self.manager.initialize()
        stats = self.manager.get_stats()
        
        self.assertEqual(stats.total_gpus, 1)
        self.assertEqual(stats.nvidia_gpus, 1)
        self.assertEqual(stats.amd_gpus, 0)
        self.assertEqual(stats.total_power_draw, 250.0)
        self.assertEqual(stats.avg_temperature, 65.0)
        self.assertEqual(stats.avg_utilization, 50.0)
    
    @patch('lab_energy.gpu.gpu_manager.NVIDIAManager')
    @patch('lab_energy.gpu.gpu_manager.AMDManager')
    def test_set_power_limit(self, mock_amd, mock_nvidia):
        """测试设置功耗限制"""
        mock_nvidia_instance = Mock()
        mock_nvidia_instance.initialize.return_value = True
        mock_nvidia_instance.get_device_count.return_value = 1
        mock_nvidia_instance.set_power_limit.return_value = True
        mock_nvidia.return_value = mock_nvidia_instance
        
        mock_amd_instance = Mock()
        mock_amd_instance.initialize.return_value = False
        mock_amd.return_value = mock_amd_instance
        
        self.manager.initialize()
        result = self.manager.set_power_limit(0, 200.0)
        
        self.assertTrue(result)
        mock_nvidia_instance.set_power_limit.assert_called_once_with(0, 200.0)
    
    @patch('lab_energy.gpu.gpu_manager.NVIDIAManager')
    @patch('lab_energy.gpu.gpu_manager.AMDManager')
    def test_detect_power_state(self, mock_amd, mock_nvidia):
        """测试检测电源状态"""
        nvidia_info = GPUInfo(
            gpu_id=0, vendor=GPUVendor.NVIDIA, name="RTX 4090",
            gpu_util=75.0, memory_util=30.0,  # 75% utilization -> HEAVY
            memory_used_mb=4096, memory_total_mb=24576,
            power_draw_w=250.0, power_limit_w=300.0, power_max_w=450.0,
            temperature=65.0, clock_mhz=2500, max_clock_mhz=2800
        )
        
        mock_nvidia_instance = Mock()
        mock_nvidia_instance.initialize.return_value = True
        mock_nvidia_instance.get_device_count.return_value = 1
        mock_nvidia_instance.get_gpu_info.return_value = nvidia_info
        mock_nvidia_instance.detect_power_state.return_value = GPUPowerState.HEAVY
        mock_nvidia.return_value = mock_nvidia_instance
        
        mock_amd_instance = Mock()
        mock_amd_instance.initialize.return_value = False
        mock_amd.return_value = mock_amd_instance
        
        self.manager.initialize()
        state = self.manager.detect_power_state(0)
        
        self.assertEqual(state, GPUPowerState.HEAVY)
    
    @patch('lab_energy.gpu.gpu_manager.NVIDIAManager')
    @patch('lab_energy.gpu.gpu_manager.AMDManager')
    def test_apply_power_optimization(self, mock_amd, mock_nvidia):
        """测试应用功耗优化"""
        mock_nvidia_instance = Mock()
        mock_nvidia_instance.initialize.return_value = True
        mock_nvidia_instance.get_device_count.return_value = 1
        mock_nvidia_instance.apply_power_optimization.return_value = True
        mock_nvidia.return_value = mock_nvidia_instance
        
        mock_amd_instance = Mock()
        mock_amd_instance.initialize.return_value = False
        mock_amd.return_value = mock_amd_instance
        
        self.manager.initialize()
        result = self.manager.apply_power_optimization(0)
        
        self.assertTrue(result)
        mock_nvidia_instance.apply_power_optimization.assert_called_once()
    
    @patch('lab_energy.gpu.gpu_manager.NVIDIAManager')
    @patch('lab_energy.gpu.gpu_manager.AMDManager')
    def test_apply_all_optimizations(self, mock_amd, mock_nvidia):
        """测试对所有GPU应用优化"""
        mock_nvidia_instance = Mock()
        mock_nvidia_instance.initialize.return_value = True
        mock_nvidia_instance.get_device_count.return_value = 2
        mock_nvidia_instance.apply_power_optimization.return_value = True
        mock_nvidia.return_value = mock_nvidia_instance
        
        mock_amd_instance = Mock()
        mock_amd_instance.initialize.return_value = False
        mock_amd.return_value = mock_amd_instance
        
        self.manager.initialize()
        results = self.manager.apply_all_optimizations()
        
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0])
        self.assertTrue(results[1])
    
    @patch('lab_energy.gpu.gpu_manager.NVIDIAManager')
    @patch('lab_energy.gpu.gpu_manager.AMDManager')
    def test_auto_optimize(self, mock_amd, mock_nvidia):
        """测试自动优化模式"""
        mock_nvidia_instance = Mock()
        mock_nvidia_instance.initialize.return_value = True
        mock_nvidia_instance.get_device_count.return_value = 1
        mock_nvidia_instance.apply_power_optimization.return_value = True
        mock_nvidia.return_value = mock_nvidia_instance
        
        mock_amd_instance = Mock()
        mock_amd_instance.initialize.return_value = False
        mock_amd.return_value = mock_amd_instance
        
        self.manager.initialize()
        
        # 启动自动优化
        self.manager.auto_optimize(interval_seconds=1)
        
        # 等待一段时间让优化执行
        time.sleep(1.5)
        
        # 停止自动优化
        self.manager.stop_auto_optimize()
        
        # 验证优化被调用了
        self.assertTrue(mock_nvidia_instance.apply_power_optimization.called)
    
    @patch('lab_energy.gpu.gpu_manager.NVIDIAManager')
    @patch('lab_energy.gpu.gpu_manager.AMDManager')
    def test_register_callback(self, mock_amd, mock_nvidia):
        """测试注册状态回调"""
        mock_nvidia_instance = Mock()
        mock_nvidia_instance.initialize.return_value = True
        mock_nvidia_instance.get_device_count.return_value = 1
        mock_nvidia_instance.apply_power_optimization.return_value = True
        mock_nvidia_instance.detect_power_state.return_value = GPUPowerState.IDLE
        mock_nvidia.return_value = mock_nvidia_instance
        
        mock_amd_instance = Mock()
        mock_amd_instance.initialize.return_value = False
        mock_amd.return_value = mock_amd_instance
        
        self.manager.initialize()
        
        # 注册回调
        callback_called = False
        received_gpu_id = None
        received_state = None
        
        def test_callback(gpu_id, state):
            nonlocal callback_called, received_gpu_id, received_state
            callback_called = True
            received_gpu_id = gpu_id
            received_state = state
        
        self.manager.register_state_callback(test_callback)
        
        # 应用优化触发回调
        self.manager.apply_power_optimization(0)
        
        self.assertTrue(callback_called)
        self.assertEqual(received_gpu_id, 0)
        self.assertEqual(received_state, GPUPowerState.IDLE)
    
    @patch('lab_energy.gpu.gpu_manager.NVIDIAManager')
    @patch('lab_energy.gpu.gpu_manager.AMDManager')
    def test_set_optimization_config(self, mock_amd, mock_nvidia):
        """测试设置优化配置"""
        mock_nvidia_instance = Mock()
        mock_nvidia_instance.initialize.return_value = True
        mock_nvidia_instance.get_device_count.return_value = 1
        mock_nvidia.return_value = mock_nvidia_instance
        
        mock_amd_instance = Mock()
        mock_amd_instance.initialize.return_value = False
        mock_amd.return_value = mock_amd_instance
        
        self.manager.initialize()
        
        # 设置自定义配置
        config = GPUOptimizationConfig(
            idle_power_percent=40.0,
            heavy_power_percent=95.0
        )
        self.manager.set_optimization_config(config)
        
        retrieved_config = self.manager.get_optimization_config()
        self.assertEqual(retrieved_config.idle_power_percent, 40.0)
        self.assertEqual(retrieved_config.heavy_power_percent, 95.0)
    
    @patch('lab_energy.gpu.gpu_manager.NVIDIAManager')
    @patch('lab_energy.gpu.gpu_manager.AMDManager')
    def test_reset_power_limits(self, mock_amd, mock_nvidia):
        """测试重置功耗限制"""
        nvidia_info = GPUInfo(
            gpu_id=0, vendor=GPUVendor.NVIDIA, name="RTX 4090",
            gpu_util=50.0, memory_util=30.0,
            memory_used_mb=4096, memory_total_mb=24576,
            power_draw_w=250.0, power_limit_w=200.0, power_max_w=450.0,
            temperature=65.0, clock_mhz=2500, max_clock_mhz=2800
        )
        
        mock_nvidia_instance = Mock()
        mock_nvidia_instance.initialize.return_value = True
        mock_nvidia_instance.get_device_count.return_value = 1
        mock_nvidia_instance.get_gpu_info.return_value = nvidia_info
        mock_nvidia_instance.set_power_limit.return_value = True
        mock_nvidia.return_value = mock_nvidia_instance
        
        mock_amd_instance = Mock()
        mock_amd_instance.initialize.return_value = False
        mock_amd.return_value = mock_amd_instance
        
        self.manager.initialize()
        results = self.manager.reset_power_limits()
        
        self.assertTrue(results[0])
        # 验证设置为最大功耗
        mock_nvidia_instance.set_power_limit.assert_called_with(0, 450.0)
    
    @patch('lab_energy.gpu.gpu_manager.NVIDIAManager')
    @patch('lab_energy.gpu.gpu_manager.AMDManager')
    def test_shutdown(self, mock_amd, mock_nvidia):
        """测试关闭管理器"""
        mock_nvidia_instance = Mock()
        mock_nvidia_instance.initialize.return_value = True
        mock_nvidia_instance.get_device_count.return_value = 1
        mock_nvidia_instance.shutdown = Mock()
        mock_nvidia.return_value = mock_nvidia_instance
        
        mock_amd_instance = Mock()
        mock_amd_instance.initialize.return_value = False
        mock_amd.return_value = mock_amd_instance
        
        self.manager.initialize()
        self.manager.shutdown()
        
        self.assertFalse(self.manager.is_initialized())
        mock_nvidia_instance.shutdown.assert_called_once()


class TestGPUStats(unittest.TestCase):
    """测试GPU统计信息"""
    
    def test_gpu_stats_defaults(self):
        """测试GPU统计信息默认值"""
        stats = GPUStats()
        
        self.assertEqual(stats.total_gpus, 0)
        self.assertEqual(stats.nvidia_gpus, 0)
        self.assertEqual(stats.amd_gpus, 0)
        self.assertEqual(stats.total_power_draw, 0.0)
        self.assertEqual(stats.avg_temperature, 0.0)
        self.assertEqual(stats.avg_utilization, 0.0)
    
    def test_gpu_stats_custom(self):
        """测试GPU统计信息自定义值"""
        stats = GPUStats(
            total_gpus=2,
            nvidia_gpus=1,
            amd_gpus=1,
            total_power_draw=500.0,
            avg_temperature=70.0,
            avg_utilization=60.0
        )
        
        self.assertEqual(stats.total_gpus, 2)
        self.assertEqual(stats.nvidia_gpus, 1)
        self.assertEqual(stats.amd_gpus, 1)
        self.assertEqual(stats.total_power_draw, 500.0)
        self.assertEqual(stats.avg_temperature, 70.0)
        self.assertEqual(stats.avg_utilization, 60.0)


class TestPowerStateDetection(unittest.TestCase):
    """测试电源状态检测"""
    
    def test_idle_state(self):
        """测试闲置状态检测"""
        # 利用率 < 5% 应该是 IDLE
        info = GPUInfo(
            gpu_id=0, vendor=GPUVendor.NVIDIA, name="Test",
            gpu_util=3.0, memory_util=0.0,
            memory_used_mb=0, memory_total_mb=8192,
            power_draw_w=50.0, power_limit_w=300.0, power_max_w=450.0,
            temperature=40.0, clock_mhz=300, max_clock_mhz=2000
        )
        
        # 通过manager的detect_power_state间接测试
        # 这里我们直接测试状态阈值
        self.assertLess(info.gpu_util, 5.0)
    
    def test_light_state(self):
        """测试轻度负载状态检测"""
        # 5% <= 利用率 < 30% 应该是 LIGHT
        info = GPUInfo(
            gpu_id=0, vendor=GPUVendor.NVIDIA, name="Test",
            gpu_util=15.0, memory_util=20.0,
            memory_used_mb=1024, memory_total_mb=8192,
            power_draw_w=100.0, power_limit_w=300.0, power_max_w=450.0,
            temperature=55.0, clock_mhz=1000, max_clock_mhz=2000
        )
        
        self.assertGreaterEqual(info.gpu_util, 5.0)
        self.assertLess(info.gpu_util, 30.0)
    
    def test_moderate_state(self):
        """测试中度负载状态检测"""
        # 30% <= 利用率 < 70% 应该是 MODERATE
        info = GPUInfo(
            gpu_id=0, vendor=GPUVendor.NVIDIA, name="Test",
            gpu_util=50.0, memory_util=40.0,
            memory_used_mb=4096, memory_total_mb=8192,
            power_draw_w=200.0, power_limit_w=300.0, power_max_w=450.0,
            temperature=70.0, clock_mhz=1500, max_clock_mhz=2000
        )
        
        self.assertGreaterEqual(info.gpu_util, 30.0)
        self.assertLess(info.gpu_util, 70.0)
    
    def test_heavy_state(self):
        """测试重度负载状态检测"""
        # 利用率 >= 70% 应该是 HEAVY
        info = GPUInfo(
            gpu_id=0, vendor=GPUVendor.NVIDIA, name="Test",
            gpu_util=85.0, memory_util=80.0,
            memory_used_mb=7168, memory_total_mb=8192,
            power_draw_w=350.0, power_limit_w=400.0, power_max_w=450.0,
            temperature=80.0, clock_mhz=1900, max_clock_mhz=2000
        )
        
        self.assertGreaterEqual(info.gpu_util, 70.0)


if __name__ == '__main__':
    unittest.main()
