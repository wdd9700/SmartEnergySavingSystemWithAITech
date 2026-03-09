#!/usr/bin/env python3
"""
环境检测模块
自动检测运行平台：Windows / Linux / Jetson Nano / Raspberry Pi
"""
import platform
import subprocess
import os
from enum import Enum
from typing import Dict, Optional
from dataclasses import dataclass


class PlatformType(Enum):
    """平台类型"""
    WINDOWS = "windows"
    LINUX_X86 = "linux_x86"
    LINUX_ARM = "linux_arm"
    JETSON_NANO = "jetson_nano"
    JETSON_XAVIER = "jetson_xavier"
    RASPBERRY_PI = "raspberry_pi"
    UNKNOWN = "unknown"


@dataclass
class PlatformInfo:
    """平台信息"""
    platform_type: PlatformType
    os_name: str
    os_version: str
    architecture: str
    cpu_count: int
    memory_gb: float
    has_gpu: bool
    gpu_name: Optional[str]
    cuda_version: Optional[str]
    jetpack_version: Optional[str]
    
    def is_jetson(self) -> bool:
        return self.platform_type in [PlatformType.JETSON_NANO, PlatformType.JETSON_XAVIER]
    
    def is_arm(self) -> bool:
        return self.platform_type in [
            PlatformType.JETSON_NANO, 
            PlatformType.JETSON_XAVIER,
            PlatformType.RASPBERRY_PI,
            PlatformType.LINUX_ARM
        ]
    
    def get_optimal_providers(self) -> list:
        """获取最优ONNX执行提供程序"""
        if self.is_jetson() and self.has_gpu:
            return ['CUDAExecutionProvider', 'CPUExecutionProvider']
        elif self.has_gpu:
            return ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            return ['CPUExecutionProvider']


class EnvironmentDetector:
    """环境检测器"""
    
    @staticmethod
    def detect() -> PlatformInfo:
        """检测当前运行环境"""
        os_name = platform.system().lower()
        arch = platform.machine().lower()
        
        # 基础信息
        cpu_count = os.cpu_count() or 1
        memory_gb = EnvironmentDetector._get_memory_gb()
        
        # 检测GPU和CUDA
        has_gpu, gpu_name, cuda_version = EnvironmentDetector._detect_gpu()
        
        # 平台特定检测
        platform_type = PlatformType.UNKNOWN
        jetpack_version = None
        
        if os_name == 'windows':
            platform_type = PlatformType.WINDOWS
        elif os_name == 'linux':
            # 检测是否为Jetson
            jetpack_version = EnvironmentDetector._detect_jetpack()
            if jetpack_version:
                if 'nano' in EnvironmentDetector._read_jetson_model().lower():
                    platform_type = PlatformType.JETSON_NANO
                else:
                    platform_type = PlatformType.JETSON_XAVIER
            # 检测是否为树莓派
            elif EnvironmentDetector._is_raspberry_pi():
                platform_type = PlatformType.RASPBERRY_PI
            # 其他ARM设备
            elif 'arm' in arch or 'aarch64' in arch:
                platform_type = PlatformType.LINUX_ARM
            else:
                platform_type = PlatformType.LINUX_X86
        
        return PlatformInfo(
            platform_type=platform_type,
            os_name=os_name,
            os_version=platform.release(),
            architecture=arch,
            cpu_count=cpu_count,
            memory_gb=memory_gb,
            has_gpu=has_gpu,
            gpu_name=gpu_name,
            cuda_version=cuda_version,
            jetpack_version=jetpack_version
        )
    
    @staticmethod
    def _get_memory_gb() -> float:
        """获取内存大小(GB)"""
        try:
            if platform.system() == 'Linux':
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if 'MemTotal' in line:
                            kb = int(line.split()[1])
                            return kb / 1024 / 1024
            import psutil
            return psutil.virtual_memory().total / 1024 / 1024 / 1024
        except:
            return 0
    
    @staticmethod
    def _detect_gpu() -> tuple:
        """检测GPU信息"""
        has_gpu = False
        gpu_name = None
        cuda_version = None
        
        # 检测CUDA
        try:
            result = subprocess.run(['nvcc', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                has_gpu = True
                for line in result.stdout.split('\n'):
                    if 'release' in line:
                        cuda_version = line.split('release')[1].split(',')[0].strip()
                        break
        except:
            pass
        
        # 检测GPU型号
        if has_gpu:
            try:
                result = subprocess.run(['nvidia-smi', '-L'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    gpu_name = result.stdout.strip()
            except:
                pass
        
        return has_gpu, gpu_name, cuda_version
    
    @staticmethod
    def _detect_jetpack() -> Optional[str]:
        """检测JetPack版本"""
        try:
            # 方法1: 读取apt列表
            result = subprocess.run(['dpkg-query', '--show', 'nvidia-jetpack'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return result.stdout.strip().split('\t')[-1]
            
            # 方法2: 读取版本文件
            if os.path.exists('/etc/nv_tegra_release'):
                with open('/etc/nv_tegra_release', 'r') as f:
                    return f.read().strip()
        except:
            pass
        return None
    
    @staticmethod
    def _read_jetson_model() -> str:
        """读取Jetson型号"""
        try:
            with open('/proc/device-tree/model', 'r') as f:
                return f.read().strip().replace('\x00', '')
        except:
            return "Unknown"
    
    @staticmethod
    def _is_raspberry_pi() -> bool:
        """检测是否为树莓派"""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                content = f.read()
                return 'Raspberry Pi' in content or 'BCM' in content
        except:
            pass
        
        try:
            return os.path.exists('/dev/gpiomem')
        except:
            return False
    
    @staticmethod
    def get_optimization_config(platform_info: PlatformInfo) -> Dict:
        """获取平台优化配置"""
        config = {
            'input_size': (640, 480),
            'target_fps': 15,
            'inference_interval': 1,
            'enable_gpu': False,
            'providers': ['CPUExecutionProvider'],
            'num_threads': 2,
            'enable_jetson_optimizations': False
        }
        
        if platform_info.is_jetson():
            config.update({
                'input_size': (640, 480),
                'target_fps': 20,
                'inference_interval': 1,
                'enable_gpu': True,
                'providers': ['CUDAExecutionProvider', 'CPUExecutionProvider'],
                'num_threads': 2,
                'enable_jetson_optimizations': True,
                'power_mode': 'MAXN'  # 或 5W, 10W
            })
        elif platform_info.platform_type == PlatformType.RASPBERRY_PI:
            config.update({
                'input_size': (416, 320),  # 降低分辨率
                'target_fps': 8,
                'inference_interval': 2,  # 跳帧
                'enable_gpu': False,
                'providers': ['CPUExecutionProvider'],
                'num_threads': 4
            })
        elif platform_info.has_gpu:
            config.update({
                'input_size': (640, 480),
                'target_fps': 25,
                'inference_interval': 1,
                'enable_gpu': True,
                'providers': ['CUDAExecutionProvider', 'CPUExecutionProvider'],
                'num_threads': 4
            })
        
        return config


# 便捷函数
def get_platform_info() -> PlatformInfo:
    """获取当前平台信息"""
    return EnvironmentDetector.detect()


def print_platform_info(info: PlatformInfo = None):
    """打印平台信息"""
    if info is None:
        info = get_platform_info()
    
    print("=" * 50)
    print("平台检测信息")
    print("=" * 50)
    print(f"平台类型: {info.platform_type.value}")
    print(f"操作系统: {info.os_name} {info.os_version}")
    print(f"架构: {info.architecture}")
    print(f"CPU核心: {info.cpu_count}")
    print(f"内存: {info.memory_gb:.1f} GB")
    print(f"GPU: {'是' if info.has_gpu else '否'} ({info.gpu_name or 'N/A'})")
    if info.cuda_version:
        print(f"CUDA: {info.cuda_version}")
    if info.jetpack_version:
        print(f"JetPack: {info.jetpack_version}")
    print("=" * 50)


if __name__ == "__main__":
    print_platform_info()
