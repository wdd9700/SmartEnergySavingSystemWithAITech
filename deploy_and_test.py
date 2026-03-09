#!/usr/bin/env python3
"""
快速部署和测试脚本
先检测环境，再运行基础测试（不需要模型）
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("="*60)
print("智能节能系统 - 快速部署测试")
print("="*60)

# 1. 环境检测
print("\n[1/4] 检测运行环境...")
from shared.environment import get_platform_info, print_platform_info

info = get_platform_info()
print_platform_info(info)

# 2. 依赖检查
print("\n[2/4] 检查依赖...")
try:
    import cv2
    print(f"✅ OpenCV: {cv2.__version__}")
except:
    print("❌ OpenCV 未安装")

try:
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
except:
    print("❌ NumPy 未安装")

try:
    import onnxruntime as ort
    print(f"✅ ONNX Runtime: {ort.__version__}")
except:
    print("❌ ONNX Runtime 未安装")

try:
    import flask
    print(f"✅ Flask: {flask.__version__}")
except:
    print("❌ Flask 未安装")

# 3. 模型检查
print("\n[3/4] 检查模型文件...")
model_path = project_root / 'models' / 'yolov8n.onnx'
if model_path.exists():
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"✅ YOLOv8模型: {size_mb:.2f} MB")
else:
    print(f"⚠️ 模型不存在: {model_path}")
    print("   运行: python models/download_models.py")

# 4. 运行基础测试
print("\n[4/4] 运行基础功能测试...")

# 测试视频增强
print("\n  测试视频增强模块...")
from corridor_light.enhancer import LowLightEnhancer
import numpy as np

enhancer = LowLightEnhancer()
test_img = np.full((480, 640, 3), 50, dtype=np.uint8)
brightness = enhancer.estimate_brightness(test_img)
print(f"    亮度检测: {brightness:.1f} (期望值~50)")

enhanced = enhancer.enhance(test_img)
enhanced_brightness = enhancer.estimate_brightness(enhanced)
print(f"    增强后亮度: {enhanced_brightness:.1f}")
print(f"    {'✅ 亮度提升正常' if enhanced_brightness > brightness else '⚠️ 亮度提升不明显'}")

# 测试控制器
print("\n  测试灯光控制器...")
from corridor_light.controller import LightController

controller = LightController(demo_mode=True)
controller.init()

# 测试开关逻辑
controller.update(True)
assert controller.light_state == True, "开灯失败"
print("    ✅ 开灯功能正常")

# 测试强制关灯
controller.force_off()
assert controller.light_state == False, "关灯失败"
print("    ✅ 关灯功能正常")

controller.cleanup()

# 测试区域管理
print("\n  测试区域管理...")
from classroom_ac.zone_manager import ZoneManager

zones = [
    {'name': 'Front', 'coords': [[0, 0], [0.5, 0], [0.5, 1], [0, 1]]},
    {'name': 'Back', 'coords': [[0.5, 0], [1, 0], [1, 1], [0.5, 1]]}
]
zone_mgr = ZoneManager(zones, frame_size=(640, 480))
print(f"    ✅ 区域初始化: {list(zone_mgr.get_zones().keys())}")

# 测试Jetson优化（如果是Jetson）
if info.is_jetson:
    print("\n  测试Jetson优化器...")
    from shared.jetson_optimizer import JetsonOptimizer
    
    optimizer = JetsonOptimizer()
    status = optimizer.get_status()
    if status:
        print(f"    当前电源模式: {status.power_mode}")
        print(f"    CPU频率: {status.cpu_freq_mhz}")
        print(f"    GPU频率: {status.gpu_freq_mhz} MHz")
        print(f"    温度: {max(status.temperature.values()):.1f}°C")

print("\n" + "="*60)
print("基础测试完成!")
print("="*60)
print(f"\n平台: {info.platform_type.value}")
print(f"推荐配置: {info.get_optimal_providers()}")

if not model_path.exists():
    print("\n⚠️  请下载模型后运行完整测试:")
    print("   cd models && python download_models.py")
else:
    print("\n✅ 可以运行完整测试:")
    print("   python tests/test_suite.py")

print("\n启动Web界面:")
print("   cd web && python server.py")
print("   然后访问 http://localhost:5000")
