#!/usr/bin/env python3
"""
Jetson Nano 优化模块
提供性能优化、电源管理、模型加速等功能
"""
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass


@dataclass
class JetsonStatus:
    """Jetson状态信息"""
    power_mode: str
    cpu_freq_mhz: List[int]
    gpu_freq_mhz: int
    emc_freq_mhz: int  # 内存频率
    temperature: Dict[str, float]  # 各传感器温度
    power_consumption: float  # 功耗(瓦特)
    ram_used_mb: int
    ram_total_mb: int


class JetsonOptimizer:
    """Jetson优化器"""
    
    POWER_MODES = {
        'MAXN': {'id': 0, 'desc': 'Maximum performance, 10W'},
        '5W': {'id': 1, 'desc': '5W power mode'},
        '10W': {'id': 2, 'desc': '10W power mode'}
    }
    
    def __init__(self):
        self.is_jetson = self._check_jetson()
        self.current_mode = None
        self.optimization_enabled = False
        self._monitor_thread = None
        self._stop_monitor = threading.Event()
        
    def _check_jetson(self) -> bool:
        """检查是否为Jetson设备"""
        return Path('/etc/nv_tegra_release').exists() or \
               Path('/proc/device-tree/model').exists()
    
    def set_power_mode(self, mode: str = 'MAXN') -> bool:
        """
        设置电源模式
        
        Args:
            mode: 'MAXN', '5W', '10W'
        """
        if not self.is_jetson:
            print("非Jetson设备，跳过电源模式设置")
            return False
        
        mode_info = self.POWER_MODES.get(mode)
        if not mode_info:
            print(f"未知电源模式: {mode}")
            return False
        
        try:
            result = subprocess.run(
                ['sudo', 'nvpmodel', '-m', str(mode_info['id'])],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                self.current_mode = mode
                print(f"电源模式已设置为: {mode} - {mode_info['desc']}")
                
                # 启用jetson_clocks以获得最佳性能
                if mode == 'MAXN':
                    self._enable_jetson_clocks()
                
                return True
            else:
                print(f"设置电源模式失败: {result.stderr}")
                return False
        except Exception as e:
            print(f"设置电源模式错误: {e}")
            return False
    
    def _enable_jetson_clocks(self):
        """启用Jetson Clocks以获得最大性能"""
        try:
            subprocess.run(['sudo', 'jetson_clocks'], 
                         capture_output=True, timeout=30)
            print("Jetson Clocks 已启用")
        except Exception as e:
            print(f"启用Jetson Clocks失败: {e}")
    
    def get_status(self) -> Optional[JetsonStatus]:
        """获取Jetson当前状态"""
        if not self.is_jetson:
            return None
        
        try:
            # 获取温度
            temps = self._get_temperatures()
            
            # 获取频率信息
            cpu_freqs = self._get_cpu_frequencies()
            gpu_freq = self._get_gpu_frequency()
            emc_freq = self._get_emc_frequency()
            
            # 获取功耗
            power = self._get_power_consumption()
            
            # 获取内存
            ram_used, ram_total = self._get_ram_info()
            
            # 获取当前电源模式
            mode = self._get_current_power_mode()
            
            return JetsonStatus(
                power_mode=mode,
                cpu_freq_mhz=cpu_freqs,
                gpu_freq_mhz=gpu_freq,
                emc_freq_mhz=emc_freq,
                temperature=temps,
                power_consumption=power,
                ram_used_mb=ram_used,
                ram_total_mb=ram_total
            )
        except Exception as e:
            print(f"获取状态失败: {e}")
            return None
    
    def _get_temperatures(self) -> Dict[str, float]:
        """获取各传感器温度"""
        temps = {}
        try:
            # 读取thermal zones
            thermal_path = Path('/sys/class/thermal')
            for zone in thermal_path.glob('thermal_zone*/'):
                try:
                    type_file = zone / 'type'
                    temp_file = zone / 'temp'
                    
                    if type_file.exists() and temp_file.exists():
                        sensor_type = type_file.read_text().strip()
                        temp = int(temp_file.read_text().strip()) / 1000.0
                        temps[sensor_type] = temp
                except:
                    continue
        except:
            pass
        return temps
    
    def _get_cpu_frequencies(self) -> List[int]:
        """获取CPU频率"""
        freqs = []
        try:
            for cpu in range(4):  # Jetson Nano有4核
                freq_file = Path(f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq')
                if freq_file.exists():
                    freq = int(freq_file.read_text().strip()) // 1000  # 转换为MHz
                    freqs.append(freq)
        except:
            pass
        return freqs
    
    def _get_gpu_frequency(self) -> int:
        """获取GPU频率"""
        try:
            freq_file = Path('/sys/devices/gpu.0/devfreq/57000000.gpu/cur_freq')
            if freq_file.exists():
                return int(freq_file.read_text().strip()) // 1000000  # 转换为MHz
        except:
            pass
        return 0
    
    def _get_emc_frequency(self) -> int:
        """获取内存控制器频率"""
        try:
            freq_file = Path('/sys/kernel/debug/bpmp/debug/clk/emc/rate')
            if freq_file.exists():
                return int(freq_file.read_text().strip()) // 1000000
        except:
            pass
        return 0
    
    def _get_power_consumption(self) -> float:
        """获取功耗"""
        try:
            # 尝试读取INA传感器
            power_file = Path('/sys/bus/i2c/drivers/ina3221x/6-0040/iio_device/in_power0_input')
            if power_file.exists():
                return float(power_file.read_text().strip()) / 1000.0
        except:
            pass
        return 0.0
    
    def _get_ram_info(self) -> tuple:
        """获取内存信息"""
        try:
            with open('/proc/meminfo', 'r') as f:
                lines = f.readlines()
                total = 0
                available = 0
                for line in lines:
                    if 'MemTotal' in line:
                        total = int(line.split()[1]) // 1024  # MB
                    elif 'MemAvailable' in line:
                        available = int(line.split()[1]) // 1024
                used = total - available
                return used, total
        except:
            pass
        return 0, 0
    
    def _get_current_power_mode(self) -> str:
        """获取当前电源模式"""
        try:
            result = subprocess.run(['nvpmodel', '-q'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'NV Power Mode' in line:
                        return line.split(':')[-1].strip()
        except:
            pass
        return 'UNKNOWN'
    
    def optimize_for_inference(self):
        """为推理任务优化系统"""
        if not self.is_jetson:
            return
        
        print("正在优化Jetson推理性能...")
        
        # 1. 设置电源模式
        self.set_power_mode('MAXN')
        
        # 2. 设置CPU调度器
        self._optimize_cpu_scheduler()
        
        # 3. 设置GPU时钟
        self._set_gpu_clock()
        
        # 4. 启动温度监控
        self._start_temperature_monitor()
        
        self.optimization_enabled = True
        print("Jetson优化完成")
    
    def _optimize_cpu_scheduler(self):
        """优化CPU调度器"""
        try:
            for cpu in range(4):
                governor_file = Path(f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor')
                if governor_file.exists():
                    subprocess.run(['sudo', 'tee', str(governor_file)], 
                                 input='performance', text=True, capture_output=True)
            print("CPU调度器已设置为performance模式")
        except Exception as e:
            print(f"CPU调度器优化失败: {e}")
    
    def _set_gpu_clock(self):
        """设置GPU时钟"""
        try:
            # Jetson Clocks会自动处理
            subprocess.run(['sudo', 'jetson_clocks', '--show'], 
                         capture_output=True, timeout=5)
        except:
            pass
    
    def _start_temperature_monitor(self):
        """启动温度监控线程"""
        def monitor():
            while not self._stop_monitor.is_set():
                status = self.get_status()
                if status and status.temperature:
                    max_temp = max(status.temperature.values())
                    if max_temp > 85:
                        print(f"⚠️ 警告: 温度过高 ({max_temp}°C)，建议降低负载")
                    elif max_temp > 80:
                        print(f"⚡ 注意: 温度较高 ({max_temp}°C)")
                self._stop_monitor.wait(10)  # 每10秒检查一次
        
        if self._monitor_thread is None:
            self._monitor_thread = threading.Thread(target=monitor, daemon=True)
            self._monitor_thread.start()
    
    def stop(self):
        """停止优化和监控"""
        self._stop_monitor.set()
        print("Jetson优化器已停止")
    
    def print_status(self):
        """打印当前状态"""
        status = self.get_status()
        if not status:
            print("无法获取Jetson状态")
            return
        
        print("=" * 50)
        print("Jetson 状态")
        print("=" * 50)
        print(f"电源模式: {status.power_mode}")
        print(f"CPU频率: {status.cpu_freq_mhz} MHz")
        print(f"GPU频率: {status.gpu_freq_mhz} MHz")
        print(f"内存频率: {status.emc_freq_mhz} MHz")
        print(f"功耗: {status.power_consumption:.2f} W")
        print(f"内存使用: {status.ram_used_mb}/{status.ram_total_mb} MB")
        print(f"\n温度:")
        for sensor, temp in status.temperature.items():
            print(f"  {sensor}: {temp:.1f}°C")
        print("=" * 50)


class TensorRTConverter:
    """TensorRT模型转换器"""
    
    def __init__(self):
        self.available = self._check_tensorrt()
    
    def _check_tensorrt(self) -> bool:
        """检查TensorRT是否可用"""
        try:
            import tensorrt as trt
            return True
        except ImportError:
            return False
    
    def convert_onnx_to_trt(self, onnx_path: str, output_path: str,
                           fp16: bool = True) -> bool:
        """
        将ONNX模型转换为TensorRT引擎
        
        Args:
            onnx_path: ONNX模型路径
            output_path: 输出引擎路径
            fp16: 是否使用FP16精度
        """
        if not self.available:
            print("TensorRT不可用，跳过转换")
            return False
        
        try:
            import tensorrt as trt
            
            logger = trt.Logger(trt.Logger.INFO)
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, logger)
            
            # 解析ONNX
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return False
            
            # 配置builder
            config = builder.create_builder_config()
            if fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            
            # Jetson Nano内存有限，需要限制workspace
            config.max_workspace_size = 1 << 30  # 1GB
            
            # 构建引擎
            print(f"正在构建TensorRT引擎: {output_path}")
            engine = builder.build_engine(network, config)
            
            if engine:
                with open(output_path, 'wb') as f:
                    f.write(engine.serialize())
                print(f"TensorRT引擎已保存: {output_path}")
                return True
            else:
                print("引擎构建失败")
                return False
        
        except Exception as e:
            print(f"转换失败: {e}")
            return False
    
    def get_engine_info(self, engine_path: str) -> Dict:
        """获取引擎信息"""
        if not self.available:
            return {}
        
        try:
            import tensorrt as trt
            
            logger = trt.Logger(trt.Logger.WARNING)
            with open(engine_path, 'rb') as f:
                runtime = trt.Runtime(logger)
                engine = runtime.deserialize_cuda_engine(f.read())
            
            return {
                'num_bindings': engine.num_bindings,
                'num_layers': engine.num_layers,
                'max_batch_size': engine.max_batch_size,
                'workspace_size': engine.workspace_size
            }
        except:
            return {}


def optimize_system():
    """一键优化系统"""
    optimizer = JetsonOptimizer()
    
    if optimizer.is_jetson:
        print("检测到Jetson设备，开始优化...")
        optimizer.optimize_for_inference()
        optimizer.print_status()
    else:
        print("非Jetson设备，跳过优化")
    
    return optimizer


if __name__ == "__main__":
    optimizer = optimize_system()
