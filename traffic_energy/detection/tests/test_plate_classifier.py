#!/usr/bin/env python3
"""
车牌分类器测试模块

测试PlateClassifier的各项功能，包括：
- HSV颜色空间分类
- RGB颜色空间分类
- 车牌区域检测
- 分类准确率
- 性能基准

Usage:
    python -m pytest traffic_energy/detection/tests/test_plate_classifier.py -v
    python traffic_energy/detection/tests/test_plate_classifier.py
"""

import unittest
import sys
import os
from pathlib import Path

import numpy as np
import cv2

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from traffic_energy.detection.plate_classifier import (
    PlateClassifier,
    PlateClassificationResult,
    PlateColor,
    PowerType,
    VehicleDetectorWithPlate
)


class TestPlateClassifier(unittest.TestCase):
    """车牌分类器测试类"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.hsv_classifier = PlateClassifier(method="hsv")
        cls.rgb_classifier = PlateClassifier(method="rgb")
    
    def _create_synthetic_plate(self, color: str, size: tuple = (200, 60)) -> np.ndarray:
        """创建合成车牌图像用于测试
        
        Args:
            color: 车牌颜色 (blue/green/yellow/white)
            size: 图像尺寸 (宽, 高)
            
        Returns:
            np.ndarray: BGR格式图像
        """
        width, height = size
        
        if color == "blue":
            # 蓝牌: B高, G中, R低
            bgr_color = (180, 100, 50)
        elif color == "green":
            # 绿牌: G高, B中, R低
            bgr_color = (80, 180, 50)
        elif color == "yellow":
            # 黄牌: R和G高, B低
            bgr_color = (50, 200, 200)
        elif color == "white":
            # 白牌: 所有通道高
            bgr_color = (220, 220, 220)
        elif color == "black":
            # 黑牌: 所有通道低
            bgr_color = (30, 30, 30)
        else:
            bgr_color = (128, 128, 128)
        
        # 创建纯色图像
        img = np.full((height, width, 3), bgr_color, dtype=np.uint8)
        
        # 添加一些噪声模拟真实场景
        noise = np.random.normal(0, 10, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return img
    
    def test_initialization(self):
        """测试分类器初始化"""
        # 测试HSV方法
        classifier_hsv = PlateClassifier(method="hsv")
        self.assertEqual(classifier_hsv.method, "hsv")
        
        # 测试RGB方法
        classifier_rgb = PlateClassifier(method="rgb")
        self.assertEqual(classifier_rgb.method, "rgb")
        
        # 测试无效方法
        with self.assertRaises(ValueError):
            PlateClassifier(method="invalid")
    
    def test_hsv_blue_classification(self):
        """测试HSV方法识别蓝牌"""
        blue_plate = self._create_synthetic_plate("blue")
        result = self.hsv_classifier.classify(blue_plate)
        
        self.assertIsInstance(result, PlateClassificationResult)
        self.assertEqual(result.color, "blue")
        self.assertEqual(result.power_type, "fuel")
        self.assertGreater(result.confidence, 0.5)
        self.assertEqual(result.method, "hsv")
    
    def test_hsv_green_classification(self):
        """测试HSV方法识别绿牌"""
        green_plate = self._create_synthetic_plate("green")
        result = self.hsv_classifier.classify(green_plate)
        
        self.assertEqual(result.color, "green")
        self.assertEqual(result.power_type, "electric")
        self.assertGreater(result.confidence, 0.5)
    
    def test_rgb_blue_classification(self):
        """测试RGB方法识别蓝牌"""
        blue_plate = self._create_synthetic_plate("blue")
        result = self.rgb_classifier.classify(blue_plate)
        
        self.assertEqual(result.color, "blue")
        self.assertEqual(result.power_type, "fuel")
        self.assertGreater(result.confidence, 0.4)
    
    def test_rgb_green_classification(self):
        """测试RGB方法识别绿牌"""
        green_plate = self._create_synthetic_plate("green")
        result = self.rgb_classifier.classify(green_plate)
        
        self.assertEqual(result.color, "green")
        self.assertEqual(result.power_type, "electric")
        self.assertGreater(result.confidence, 0.4)
    
    def test_empty_image(self):
        """测试空图像处理"""
        empty_img = np.array([])
        result = self.hsv_classifier.classify(empty_img)
        
        self.assertEqual(result.color, "unknown")
        self.assertEqual(result.power_type, "unknown")
        self.assertEqual(result.confidence, 0.0)
    
    def test_none_input(self):
        """测试None输入处理"""
        result = self.hsv_classifier.classify(None)
        
        self.assertEqual(result.color, "unknown")
        self.assertEqual(result.power_type, "unknown")
        self.assertEqual(result.confidence, 0.0)
    
    def test_yellow_plate(self):
        """测试黄牌识别"""
        yellow_plate = self._create_synthetic_plate("yellow")
        result = self.hsv_classifier.classify(yellow_plate)
        
        # 黄牌应该被识别为fuel类型
        self.assertEqual(result.color, "yellow")
        self.assertEqual(result.power_type, "fuel")
    
    def test_white_plate(self):
        """测试白牌识别"""
        white_plate = self._create_synthetic_plate("white")
        result = self.hsv_classifier.classify(white_plate)
        
        self.assertEqual(result.color, "white")
    
    def test_black_plate(self):
        """测试黑牌识别"""
        black_plate = self._create_synthetic_plate("black")
        result = self.hsv_classifier.classify(black_plate)
        
        self.assertEqual(result.color, "black")
    
    def test_performance_benchmark(self):
        """测试性能基准"""
        blue_plate = self._create_synthetic_plate("blue")
        
        # 预热
        for _ in range(10):
            self.hsv_classifier.classify(blue_plate)
        
        # 重置统计
        self.hsv_classifier.reset_stats()
        
        # 运行100次测试
        num_iterations = 100
        for _ in range(num_iterations):
            self.hsv_classifier.classify(blue_plate)
        
        avg_time = self.hsv_classifier.average_processing_time
        
        # 断言处理时间小于5ms
        self.assertLess(avg_time, 5.0, 
                       f"平均处理时间 {avg_time:.2f}ms 超过5ms阈值")
        
        print(f"\n性能测试结果:")
        print(f"  平均处理时间: {avg_time:.3f}ms")
        print(f"  总处理次数: {self.hsv_classifier.classification_count}")
    
    def test_plate_region_detection(self):
        """测试车牌区域检测"""
        # 创建一个模拟车辆图像，车牌在下半部分
        vehicle_height, vehicle_width = 400, 600
        vehicle_img = np.full((vehicle_height, vehicle_width, 3), (100, 100, 100), dtype=np.uint8)
        
        # 在底部添加绿色车牌区域
        plate_y_start = int(vehicle_height * 0.75)
        plate_y_end = int(vehicle_height * 0.9)
        plate_x_start = int(vehicle_width * 0.3)
        plate_x_end = int(vehicle_width * 0.7)
        
        vehicle_img[plate_y_start:plate_y_end, plate_x_start:plate_x_end] = (80, 180, 50)
        
        # 检测车牌区域
        detected_plate = self.hsv_classifier.detect_plate_region(vehicle_img)
        
        self.assertIsNotNone(detected_plate)
        self.assertGreater(detected_plate.shape[0], 0)
        self.assertGreater(detected_plate.shape[1], 0)
    
    def test_batch_classification(self):
        """测试批量分类"""
        plates = [
            self._create_synthetic_plate("blue"),
            self._create_synthetic_plate("green"),
            self._create_synthetic_plate("blue"),
        ]
        
        results = self.hsv_classifier.classify_batch(plates)
        
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].color, "blue")
        self.assertEqual(results[1].color, "green")
        self.assertEqual(results[2].color, "blue")
    
    def test_stats_tracking(self):
        """测试统计信息追踪"""
        classifier = PlateClassifier(method="hsv")
        
        # 初始状态
        self.assertEqual(classifier.classification_count, 0)
        self.assertEqual(classifier.average_processing_time, 0.0)
        
        # 执行分类
        plate = self._create_synthetic_plate("blue")
        classifier.classify(plate)
        
        self.assertEqual(classifier.classification_count, 1)
        self.assertGreater(classifier.average_processing_time, 0)
        
        # 获取统计信息
        stats = classifier.get_stats()
        self.assertIn("classification_count", stats)
        self.assertIn("average_processing_time_ms", stats)
        self.assertIn("method", stats)
        
        # 重置统计
        classifier.reset_stats()
        self.assertEqual(classifier.classification_count, 0)


class TestVehicleDetectorWithPlate(unittest.TestCase):
    """集成检测器测试类"""
    
    def test_initialization_without_plate(self):
        """测试不启用车牌识别的初始化"""
        # 注意：这需要YOLO模型，测试中跳过实际检测
        # 仅测试初始化逻辑
        pass


class TestAccuracyBenchmark(unittest.TestCase):
    """准确率基准测试"""
    
    def test_blue_plate_accuracy(self):
        """测试蓝牌识别准确率"""
        classifier = PlateClassifier(method="hsv")
        
        correct = 0
        total = 50
        
        for _ in range(total):
            # 创建带变异的蓝牌
            plate = self._create_blue_plate_with_variation()
            result = classifier.classify(plate)
            
            if result.color == "blue" and result.power_type == "fuel":
                correct += 1
        
        accuracy = correct / total
        print(f"\n蓝牌识别准确率: {accuracy*100:.1f}%")
        self.assertGreaterEqual(accuracy, 0.95, f"蓝牌准确率 {accuracy*100:.1f}% 低于95%")
    
    def test_green_plate_accuracy(self):
        """测试绿牌识别准确率"""
        classifier = PlateClassifier(method="hsv")
        
        correct = 0
        total = 50
        
        for _ in range(total):
            # 创建带变异的绿牌
            plate = self._create_green_plate_with_variation()
            result = classifier.classify(plate)
            
            if result.color == "green" and result.power_type == "electric":
                correct += 1
        
        accuracy = correct / total
        print(f"\n绿牌识别准确率: {accuracy*100:.1f}%")
        self.assertGreaterEqual(accuracy, 0.95, f"绿牌准确率 {accuracy*100:.1f}% 低于95%")
    
    def _create_blue_plate_with_variation(self) -> np.ndarray:
        """创建带光照变异的蓝牌"""
        width, height = 200, 60
        
        # 基础蓝色，添加随机变异
        base_b = np.random.randint(160, 200)
        base_g = np.random.randint(80, 120)
        base_r = np.random.randint(30, 70)
        
        img = np.full((height, width, 3), (base_b, base_g, base_r), dtype=np.uint8)
        
        # 添加噪声
        noise = np.random.normal(0, 15, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return img
    
    def _create_green_plate_with_variation(self) -> np.ndarray:
        """创建带光照变异的绿牌"""
        width, height = 200, 60
        
        # 基础绿色，添加随机变异
        base_b = np.random.randint(60, 100)
        base_g = np.random.randint(160, 200)
        base_r = np.random.randint(30, 70)
        
        img = np.full((height, width, 3), (base_b, base_g, base_r), dtype=np.uint8)
        
        # 添加噪声
        noise = np.random.normal(0, 15, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return img


def run_visual_test():
    """运行可视化测试（可选）"""
    print("\n" + "="*50)
    print("可视化测试")
    print("="*50)
    
    classifier = PlateClassifier(method="hsv")
    
    # 测试不同颜色
    colors = ["blue", "green", "yellow", "white", "black"]
    
    for color in colors:
        # 创建合成车牌
        if color == "blue":
            img = np.full((60, 200, 3), (180, 100, 50), dtype=np.uint8)
        elif color == "green":
            img = np.full((60, 200, 3), (80, 180, 50), dtype=np.uint8)
        elif color == "yellow":
            img = np.full((60, 200, 3), (50, 200, 200), dtype=np.uint8)
        elif color == "white":
            img = np.full((60, 200, 3), (220, 220, 220), dtype=np.uint8)
        else:
            img = np.full((60, 200, 3), (30, 30, 30), dtype=np.uint8)
        
        result = classifier.classify(img)
        
        print(f"\n{color.upper()} 车牌:")
        print(f"  识别颜色: {result.color}")
        print(f"  动力类型: {result.power_type}")
        print(f"  置信度: {result.confidence:.2f}")
        print(f"  处理时间: {result.processing_time_ms:.2f}ms")


if __name__ == "__main__":
    # 运行可视化测试
    run_visual_test()
    
    # 运行单元测试
    print("\n" + "="*50)
    print("运行单元测试")
    print("="*50)
    
    unittest.main(verbosity=2, exit=False)
