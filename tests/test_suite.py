#!/usr/bin/env python3
"""
自动化测试脚本
测试智能节能系统的各个模块

测试内容:
1. 环境检测
2. 模型加载和推理
3. 视频增强算法
4. 检测准确率评估
5. 控制器逻辑
6. 性能基准测试
"""
import sys
import time
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from shared.environment import EnvironmentDetector, get_platform_info, print_platform_info
from shared.performance import PerformanceMonitor
from corridor_light.detector import PersonDetector
from corridor_light.enhancer import LowLightEnhancer
from corridor_light.controller import LightController
from classroom_ac.people_counter import PeopleCounter
from classroom_ac.zone_manager import ZoneManager


class TestResult:
    """测试结果"""
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.duration_ms = 0
        self.details = {}
        self.error = None
    
    def to_dict(self):
        return {
            'name': self.name,
            'passed': self.passed,
            'duration_ms': self.duration_ms,
            'details': self.details,
            'error': str(self.error) if self.error else None
        }


class TestSuite:
    """测试套件"""
    
    def __init__(self):
        self.results = []
        self.start_time = None
        
    def run_test(self, name: str, test_func) -> TestResult:
        """运行单个测试"""
        print(f"\n{'='*60}")
        print(f"测试: {name}")
        print('='*60)
        
        result = TestResult(name)
        start = time.time()
        
        try:
            test_func(result)
            result.passed = True
            print(f"✅ 通过")
        except Exception as e:
            result.error = e
            print(f"❌ 失败: {e}")
        
        result.duration_ms = (time.time() - start) * 1000
        self.results.append(result)
        
        return result
    
    def generate_report(self) -> dict:
        """生成测试报告"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        
        return {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total': total,
                'passed': passed,
                'failed': failed,
                'pass_rate': passed / total if total > 0 else 0
            },
            'tests': [r.to_dict() for r in self.results]
        }


def test_environment_detection(result: TestResult):
    """测试环境检测"""
    print("检测运行环境...")
    
    info = get_platform_info()
    print_platform_info(info)
    
    result.details['platform'] = info.platform_type.value
    result.details['has_gpu'] = info.has_gpu
    result.details['cuda_version'] = info.cuda_version
    result.details['memory_gb'] = info.memory_gb
    
    # 获取优化配置
    config = EnvironmentDetector.get_optimization_config(info)
    result.details['optimization'] = config
    
    print(f"优化配置: {json.dumps(config, indent=2)}")


def test_model_loading(result: TestResult):
    """测试模型加载"""
    print("测试YOLOv8模型加载...")
    
    model_path = project_root / 'models' / 'yolov8n.onnx'
    
    if not model_path.exists():
        result.error = Exception(f"模型文件不存在: {model_path}")
        return
    
    detector = PersonDetector(str(model_path))
    success = detector.load_model()
    
    if not success:
        result.error = Exception("模型加载失败")
        return
    
    result.details['model_path'] = str(model_path)
    result.details['input_shape'] = detector.input_shape
    result.details['model_size_mb'] = model_path.stat().st_size / (1024 * 1024)
    
    print(f"模型大小: {result.details['model_size_mb']:.2f} MB")
    print(f"输入尺寸: {detector.input_shape}")


def test_inference_accuracy(result: TestResult):
    """测试推理准确率"""
    print("测试人形检测准确率...")
    
    model_path = project_root / 'models' / 'yolov8n.onnx'
    detector = PersonDetector(str(model_path))
    
    if not detector.load_model():
        result.error = Exception("模型加载失败")
        return
    
    # 创建测试图像
    test_cases = [
        {
            'name': '单人场景',
            'image': create_test_image_single_person(),
            'expected_min': 1,
            'expected_max': 1
        },
        {
            'name': '多人场景',
            'image': create_test_image_multiple_people(),
            'expected_min': 3,
            'expected_max': 5
        },
        {
            'name': '空场景',
            'image': create_test_image_empty(),
            'expected_min': 0,
            'expected_max': 0
        },
        {
            'name': '低光照场景',
            'image': create_test_image_low_light(),
            'expected_min': 1,
            'expected_max': 2
        }
    ]
    
    accuracy_results = []
    
    for case in test_cases:
        print(f"  测试: {case['name']}")
        
        detections = detector.detect(case['image'])
        people_count = len([d for d in detections if d['class'] == 'person'])
        
        passed = case['expected_min'] <= people_count <= case['expected_max']
        
        accuracy_results.append({
            'case': case['name'],
            'detected': people_count,
            'expected': f"{case['expected_min']}-{case['expected_max']}",
            'passed': passed
        })
        
        status = "✅" if passed else "⚠️"
        print(f"    {status} 检测到 {people_count} 人, 期望 {case['expected_min']}-{case['expected_max']}")
    
    result.details['test_cases'] = accuracy_results
    result.details['accuracy'] = sum(1 for r in accuracy_results if r['passed']) / len(accuracy_results)


def test_video_enhancement(result: TestResult):
    """测试视频增强算法"""
    print("测试低光照增强算法...")
    
    # 创建测试图像
    dark_image = create_test_image_low_light()
    
    methods = ['clahe', 'gamma', 'msrcr']
    enhancer_results = []
    
    for method in methods:
        print(f"  测试: {method.upper()}")
        
        enhancer = LowLightEnhancer(method=method, brightness_threshold=100)
        
        start = time.time()
        enhanced = enhancer.enhance(dark_image)
        duration = (time.time() - start) * 1000
        
        # 检查亮度是否提升
        original_brightness = enhancer.estimate_brightness(dark_image)
        enhanced_brightness = enhancer.estimate_brightness(enhanced)
        
        improvement = enhanced_brightness - original_brightness
        
        enhancer_results.append({
            'method': method,
            'original_brightness': original_brightness,
            'enhanced_brightness': enhanced_brightness,
            'improvement': improvement,
            'duration_ms': duration
        })
        
        print(f"    亮度: {original_brightness:.1f} -> {enhanced_brightness:.1f} (+{improvement:.1f})")
        print(f"    耗时: {duration:.2f}ms")
    
    result.details['enhancement_methods'] = enhancer_results
    result.details['best_method'] = max(enhancer_results, key=lambda x: x['improvement'])['method']


def test_controller_logic(result: TestResult):
    """测试控制器逻辑"""
    print("测试灯光控制器逻辑...")
    
    controller = LightController(
        light_on_delay=0,
        light_off_delay=2.0,
        demo_mode=True
    )
    controller.init()
    
    # 测试用例
    test_sequence = [
        {'detected': True, 'expected': True, 'desc': '检测到 -> 开灯'},
        {'detected': False, 'wait': 0.5, 'expected': True, 'desc': '消失0.5s -> 仍开'},
        {'detected': False, 'wait': 2.5, 'expected': False, 'desc': '消失2.5s -> 关灯'},
        {'detected': True, 'expected': True, 'desc': '再次检测 -> 开灯'},
    ]
    
    logic_results = []
    
    for test in test_sequence:
        if test.get('wait'):
            time.sleep(test['wait'])
        
        state = controller.update(test['detected'])
        passed = state == test['expected']
        
        logic_results.append({
            'description': test['desc'],
            'expected': test['expected'],
            'actual': state,
            'passed': passed
        })
        
        status = "✅" if passed else "❌"
        print(f"  {status} {test['desc']}: {state}")
    
    controller.cleanup()
    
    result.details['logic_tests'] = logic_results
    result.details['logic_accuracy'] = sum(1 for r in logic_results if r['passed']) / len(logic_results)


def test_performance_benchmark(result: TestResult):
    """性能基准测试"""
    print("运行性能基准测试...")
    
    model_path = project_root / 'models' / 'yolov8n.onnx'
    detector = PersonDetector(str(model_path))
    
    if not detector.load_model():
        result.error = Exception("模型加载失败")
        return
    
    # 创建测试图像
    test_image = create_test_image_multiple_people()
    
    # 预热
    for _ in range(5):
        detector.detect(test_image)
    
    # 正式测试
    iterations = 50
    times = []
    
    print(f"  运行 {iterations} 次推理...")
    
    for i in range(iterations):
        start = time.time()
        detector.detect(test_image)
        duration = (time.time() - start) * 1000
        times.append(duration)
    
    result.details['iterations'] = iterations
    result.details['avg_inference_ms'] = np.mean(times)
    result.details['min_inference_ms'] = np.min(times)
    result.details['max_inference_ms'] = np.max(times)
    result.details['std_inference_ms'] = np.std(times)
    result.details['estimated_fps'] = 1000 / np.mean(times)
    
    print(f"  平均推理时间: {result.details['avg_inference_ms']:.2f}ms")
    print(f"  最小/最大: {result.details['min_inference_ms']:.2f}ms / {result.details['max_inference_ms']:.2f}ms")
    print(f"  标准差: {result.details['std_inference_ms']:.2f}ms")
    print(f"  估算FPS: {result.details['estimated_fps']:.1f}")


def test_zone_management(result: TestResult):
    """测试区域管理"""
    print("测试区域管理...")
    
    zones = [
        {'name': 'Front', 'coords': [[0, 0], [0.5, 0], [0.5, 1], [0, 1]]},
        {'name': 'Back', 'coords': [[0.5, 0], [1, 0], [1, 1], [0.5, 1]]}
    ]
    
    zone_manager = ZoneManager(zones, frame_size=(640, 480))
    
    # 测试区域检测
    test_detections = [
        {'bbox': [100, 100, 150, 200], 'class': 'person'},  # 应在Front
        {'bbox': [400, 100, 450, 200], 'class': 'person'},  # 应在Back
    ]
    
    zone_counts = {}
    for zone_name in zone_manager.get_zones().keys():
        zone_counts[zone_name] = zone_manager.get_zone_count(zone_name, test_detections)
    
    result.details['zones'] = list(zone_manager.get_zones().keys())
    result.details['zone_counts'] = zone_counts
    
    print(f"  区域: {result.details['zones']}")
    print(f"  各区域人数: {zone_counts}")


# ==================== 测试图像生成 ====================

def create_test_image_single_person(size=(640, 480)):
    """创建单人测试图像"""
    img = np.full((*size[::-1], 3), 128, dtype=np.uint8)
    
    # 绘制简化的"人形"
    cx, cy = size[0] // 2, size[1] // 2
    cv2.ellipse(img, (cx, cy - 40), (20, 20), 0, 0, 360, (80, 80, 80), -1)
    cv2.rectangle(img, (cx - 25, cy - 20), (cx + 25, cy + 60), (60, 70, 80), -1)
    
    return img


def create_test_image_multiple_people(size=(640, 480)):
    """创建多人测试图像"""
    img = np.full((*size[::-1], 3), 128, dtype=np.uint8)
    
    positions = [(150, 240), (320, 200), (450, 260), (250, 300)]
    
    for cx, cy in positions:
        cv2.ellipse(img, (cx, cy - 40), (20, 20), 0, 0, 360, (80, 80, 80), -1)
        cv2.rectangle(img, (cx - 25, cy - 20), (cx + 25, cy + 60), (60, 70, 80), -1)
    
    return img


def create_test_image_empty(size=(640, 480)):
    """创建空场景测试图像"""
    return np.full((*size[::-1], 3), 128, dtype=np.uint8)


def create_test_image_low_light(size=(640, 480)):
    """创建低光照测试图像"""
    img = np.full((*size[::-1], 3), 30, dtype=np.uint8)
    
    # 添加暗色人形
    cx, cy = size[0] // 2, size[1] // 2
    cv2.ellipse(img, (cx, cy - 40), (20, 20), 0, 0, 360, (50, 50, 50), -1)
    cv2.rectangle(img, (cx - 25, cy - 20), (cx + 25, cy + 60), (40, 45, 50), -1)
    
    # 添加噪点
    noise = np.random.normal(0, 5, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img


# ==================== 主函数 ====================

def main():
    print("="*60)
    print("智能节能系统 - 自动化测试")
    print("="*60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    suite = TestSuite()
    
    # 运行所有测试
    suite.run_test("环境检测", test_environment_detection)
    suite.run_test("模型加载", test_model_loading)
    suite.run_test("推理准确率", test_inference_accuracy)
    suite.run_test("视频增强", test_video_enhancement)
    suite.run_test("控制器逻辑", test_controller_logic)
    suite.run_test("性能基准", test_performance_benchmark)
    suite.run_test("区域管理", test_zone_management)
    
    # 生成报告
    report = suite.generate_report()
    
    # 保存报告
    report_path = project_root / 'test_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # 打印摘要
    print("\n" + "="*60)
    print("测试摘要")
    print("="*60)
    print(f"总测试数: {report['summary']['total']}")
    print(f"通过: {report['summary']['passed']} ✅")
    print(f"失败: {report['summary']['failed']} ❌")
    print(f"通过率: {report['summary']['pass_rate']*100:.1f}%")
    print("="*60)
    
    print(f"\n详细报告已保存: {report_path}")
    
    return report


if __name__ == "__main__":
    main()
