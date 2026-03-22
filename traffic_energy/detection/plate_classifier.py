#!/usr/bin/env python3
"""
车牌颜色识别模块

基于传统CV算法（HSV/RGB颜色空间分析）实现蓝牌(燃油车)与绿牌(电动车)的分类。
计算量极小，适合边缘设备部署。

Example:
    >>> from traffic_energy.detection.plate_classifier import PlateClassifier
    >>> classifier = PlateClassifier(method="hsv")
    >>> result = classifier.classify(plate_img)
    >>> print(f"车牌类型: {result['power_type']}, 置信度: {result['confidence']:.2f}")
"""

import time
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np
import cv2

try:
    from shared.logger import setup_logger
    logger = setup_logger("plate_classifier")
except ImportError:
    import logging
    logger = logging.getLogger("plate_classifier")


class PlateColor(Enum):
    """车牌颜色枚举"""
    BLUE = "blue"      # 蓝牌 - 燃油车
    GREEN = "green"    # 绿牌 - 电动车
    YELLOW = "yellow"  # 黄牌 - 大型车辆
    WHITE = "white"    # 白牌 - 军警车辆
    BLACK = "black"    # 黑牌 - 外籍车辆
    UNKNOWN = "unknown"


class PowerType(Enum):
    """车辆动力类型枚举"""
    FUEL = "fuel"       # 燃油车
    ELECTRIC = "electric"  # 电动车
    UNKNOWN = "unknown"


@dataclass
class PlateClassificationResult:
    """车牌分类结果数据类
    
    Attributes:
        color: 车牌颜色 (blue/green/yellow/white/black/unknown)
        power_type: 车辆动力类型 (fuel/electric/unknown)
        confidence: 置信度分数 (0-1)
        method: 分类方法 (hsv/rgb)
        plate_bbox: 车牌在图像中的位置 [x1, y1, x2, y2]
        processing_time_ms: 处理时间（毫秒）
    """
    color: str
    power_type: str
    confidence: float
    method: str
    plate_bbox: Optional[np.ndarray] = None
    processing_time_ms: float = 0.0


class PlateClassifier:
    """车牌颜色分类器
    
    使用HSV或RGB颜色空间分析识别中国车牌颜色，
    区分燃油车(蓝牌)与电动车(绿牌)。
    
    Attributes:
        method: 分类方法 ("hsv" | "rgb")
        hsv_thresholds: HSV颜色阈值配置
        
    Example:
        >>> classifier = PlateClassifier(method="hsv")
        >>> result = classifier.classify(plate_img)
        >>> if result.power_type == "electric":
        ...     print("电动车 detected")
    """
    
    # HSV颜色阈值配置 (OpenCV H范围: 0-180)
    # 蓝色: H 100-140 (OpenCV中蓝色在100-140范围)
    # 绿色: H 35-85 (OpenCV中绿色在35-85范围)
    DEFAULT_HSV_THRESHOLDS = {
        "blue": {
            "h_min": 100, "h_max": 140,
            "s_min": 60, "s_max": 255,
            "v_min": 80, "v_max": 255
        },
        "green": {
            "h_min": 35, "h_max": 85,
            "s_min": 60, "s_max": 255,
            "v_min": 80, "v_max": 255
        },
        "yellow": {
            "h_min": 20, "h_max": 35,
            "s_min": 100, "s_max": 255,
            "v_min": 100, "v_max": 255
        }
    }
    
    # RGB比值阈值
    RGB_THRESHOLDS = {
        "blue": {"b_g_ratio_min": 1.1, "b_r_ratio_min": 1.2},
        "green": {"g_b_ratio_min": 1.0, "g_r_ratio_min": 1.2}
    }
    
    def __init__(
        self,
        method: str = "hsv",
        hsv_thresholds: Optional[Dict] = None,
        min_plate_size: Tuple[int, int] = (60, 20),
        confidence_threshold: float = 0.6
    ):
        """初始化车牌分类器
        
        Args:
            method: 分类方法 ("hsv" | "rgb")
            hsv_thresholds: 自定义HSV阈值配置
            min_plate_size: 最小车牌尺寸 (宽, 高)
            confidence_threshold: 置信度阈值
            
        Raises:
            ValueError: 不支持的分类方法
        """
        if method not in ["hsv", "rgb"]:
            raise ValueError(f"不支持的分类方法: {method}，请使用 'hsv' 或 'rgb'")
        
        self.method = method
        self.hsv_thresholds = hsv_thresholds or self.DEFAULT_HSV_THRESHOLDS.copy()
        self.min_plate_size = min_plate_size
        self.confidence_threshold = confidence_threshold
        
        # 性能统计
        self.classification_count = 0
        self.total_processing_time = 0.0
        
        logger.info(f"车牌分类器初始化完成，方法: {method}")
    
    def detect_plate_region(
        self,
        vehicle_img: np.ndarray,
        vehicle_bbox: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """检测车牌区域
        
        基于颜色阈值分割在车辆图像中定位车牌位置。
        
        Args:
            vehicle_img: 车辆图像 (BGR格式)
            vehicle_bbox: 车辆边界框 [x1, y1, x2, y2]（可选）
            
        Returns:
            np.ndarray: 车牌区域图像，未检测到则返回None
            
        Note:
            车牌通常位于车辆前部/后部的下半区域
        """
        if vehicle_img is None or vehicle_img.size == 0:
            return None
        
        h, w = vehicle_img.shape[:2]
        
        # 如果图像太小，直接返回
        if h < self.min_plate_size[1] or w < self.min_plate_size[0]:
            return None
        
        # 车牌通常位于车辆下半区域的中间位置
        # 裁剪下半区域进行搜索
        roi_y_start = int(h * 0.6)
        roi_y_end = int(h * 0.95)
        roi_x_start = int(w * 0.15)
        roi_x_end = int(w * 0.85)
        
        roi = vehicle_img[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        
        if roi.size == 0:
            return None
        
        # 转换为HSV进行颜色分割
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 创建蓝色和绿色的掩码
        blue_mask = self._create_color_mask(hsv_roi, "blue")
        green_mask = self._create_color_mask(hsv_roi, "green")
        
        # 合并掩码
        combined_mask = cv2.bitwise_or(blue_mask, green_mask)
        
        # 形态学操作优化掩码
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(
            combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        best_plate = None
        best_score = 0
        
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            
            # 过滤不符合车牌比例的轮廓
            aspect_ratio = cw / max(ch, 1)
            if not (2.0 <= aspect_ratio <= 5.0):
                continue
            
            # 过滤太小的区域
            if cw < self.min_plate_size[0] or ch < self.min_plate_size[1]:
                continue
            
            # 计算区域得分（颜色纯度）
            plate_roi = roi[y:y+ch, x:x+cw]
            if plate_roi.size == 0:
                continue
            
            plate_hsv = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2HSV)
            blue_pixels = cv2.countNonZero(self._create_color_mask(plate_hsv, "blue"))
            green_pixels = cv2.countNonZero(self._create_color_mask(plate_hsv, "green"))
            total_pixels = cw * ch
            
            color_ratio = max(blue_pixels, green_pixels) / max(total_pixels, 1)
            
            if color_ratio > best_score:
                best_score = color_ratio
                best_plate = plate_roi.copy()
        
        # 如果没有检测到合适的车牌区域，返回整个ROI区域
        if best_plate is None and roi.size > 0:
            # 检查ROI中是否有足够的颜色信息
            hsv_full = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            blue_mask_full = self._create_color_mask(hsv_full, "blue")
            green_mask_full = self._create_color_mask(hsv_full, "green")
            
            total_pixels = roi.shape[0] * roi.shape[1]
            blue_ratio = cv2.countNonZero(blue_mask_full) / total_pixels
            green_ratio = cv2.countNonZero(green_mask_full) / total_pixels
            
            if blue_ratio > 0.1 or green_ratio > 0.1:
                best_plate = roi
        
        return best_plate
    
    def _create_color_mask(
        self,
        hsv_img: np.ndarray,
        color: str
    ) -> np.ndarray:
        """创建颜色掩码
        
        Args:
            hsv_img: HSV格式图像
            color: 颜色名称 (blue/green/yellow)
            
        Returns:
            np.ndarray: 二值掩码图像
        """
        thresholds = self.hsv_thresholds.get(color, {})
        
        lower = np.array([
            thresholds.get("h_min", 0),
            thresholds.get("s_min", 0),
            thresholds.get("v_min", 0)
        ])
        upper = np.array([
            thresholds.get("h_max", 180),
            thresholds.get("s_max", 255),
            thresholds.get("v_max", 255)
        ])
        
        return cv2.inRange(hsv_img, lower, upper)
    
    def classify(
        self,
        plate_img: np.ndarray,
        detect_region: bool = False,
        vehicle_img: Optional[np.ndarray] = None,
        vehicle_bbox: Optional[np.ndarray] = None
    ) -> PlateClassificationResult:
        """分类车牌颜色
        
        Args:
            plate_img: 车牌图像 (BGR格式)
            detect_region: 是否需要先检测车牌区域
            vehicle_img: 车辆图像（detect_region为True时需要）
            vehicle_bbox: 车辆边界框（detect_region为True时需要）
            
        Returns:
            PlateClassificationResult: 分类结果
            
        Example:
            >>> result = classifier.classify(plate_img)
            >>> print(f"颜色: {result.color}, 类型: {result.power_type}")
        """
        start_time = time.time()
        
        # 如果需要，先检测车牌区域
        if detect_region and vehicle_img is not None:
            detected_plate = self.detect_plate_region(vehicle_img, vehicle_bbox)
            if detected_plate is not None:
                plate_img = detected_plate
        
        # 验证输入
        if plate_img is None or plate_img.size == 0:
            processing_time = (time.time() - start_time) * 1000
            return PlateClassificationResult(
                color="unknown",
                power_type="unknown",
                confidence=0.0,
                method=self.method,
                processing_time_ms=processing_time
            )
        
        # 预处理：调整大小以提高稳定性
        plate_img = self._preprocess_plate(plate_img)
        
        # 根据方法进行分类
        if self.method == "hsv":
            color, power_type, confidence = self._classify_hsv(plate_img)
        else:
            color, power_type, confidence = self._classify_rgb(plate_img)
        
        processing_time = (time.time() - start_time) * 1000
        
        # 更新统计
        self.classification_count += 1
        self.total_processing_time += processing_time
        
        return PlateClassificationResult(
            color=color,
            power_type=power_type,
            confidence=confidence,
            method=self.method,
            processing_time_ms=processing_time
        )
    
    def _preprocess_plate(self, plate_img: np.ndarray) -> np.ndarray:
        """预处理车牌图像
        
        Args:
            plate_img: 原始车牌图像
            
        Returns:
            np.ndarray: 预处理后的图像
        """
        # 调整大小到标准尺寸以提高稳定性
        target_width = 200
        h, w = plate_img.shape[:2]
        aspect_ratio = h / w
        target_height = int(target_width * aspect_ratio)
        
        # 限制高度范围
        target_height = max(40, min(target_height, 80))
        
        resized = cv2.resize(plate_img, (target_width, target_height))
        
        return resized
    
    def _classify_hsv(
        self,
        plate_img: np.ndarray
    ) -> Tuple[str, str, float]:
        """基于HSV颜色空间分类
        
        Args:
            plate_img: 车牌图像 (BGR格式)
            
        Returns:
            Tuple[str, str, float]: (颜色, 动力类型, 置信度)
        """
        # 转换到HSV
        hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)
        
        # 提取底色区域（去除边缘和文字区域）
        h, w = hsv.shape[:2]
        
        # 取中间区域，避免边缘干扰
        roi = hsv[int(h*0.2):int(h*0.8), int(w*0.1):int(w*0.9)]
        
        if roi.size == 0:
            return "unknown", "unknown", 0.0
        
        # 计算HSV统计值
        mean_h = np.mean(roi[:, :, 0])
        mean_s = np.mean(roi[:, :, 1])
        mean_v = np.mean(roi[:, :, 2])
        
        # 创建颜色掩码统计像素分布
        blue_mask = self._create_color_mask(roi, "blue")
        green_mask = self._create_color_mask(roi, "green")
        yellow_mask = self._create_color_mask(roi, "yellow")
        
        total_pixels = roi.shape[0] * roi.shape[1]
        blue_ratio = cv2.countNonZero(blue_mask) / total_pixels
        green_ratio = cv2.countNonZero(green_mask) / total_pixels
        yellow_ratio = cv2.countNonZero(yellow_mask) / total_pixels
        
        # 分类逻辑
        # 首先检查饱和度是否足够（避免灰度图像干扰）
        if mean_s < 40:
            # 低饱和度，可能是白牌或黑牌
            if mean_v > 180:
                return "white", "unknown", 0.5
            elif mean_v < 80:
                return "black", "unknown", 0.5
            return "unknown", "unknown", 0.3
        
        # 根据颜色像素比例和色相分类
        max_ratio = max(blue_ratio, green_ratio, yellow_ratio)
        
        if max_ratio < 0.15:
            # 颜色像素不足，可能是其他类型车牌
            return "unknown", "unknown", 0.3
        
        # 判断主要颜色
        if green_ratio >= blue_ratio and green_ratio >= yellow_ratio:
            # 绿色车牌 - 电动车
            confidence = min(0.95, 0.6 + green_ratio * 0.5)
            return "green", "electric", confidence
        elif blue_ratio >= green_ratio and blue_ratio >= yellow_ratio:
            # 蓝色车牌 - 燃油车
            confidence = min(0.95, 0.6 + blue_ratio * 0.5)
            return "blue", "fuel", confidence
        elif yellow_ratio > 0.3:
            # 黄色车牌 - 大型车辆（通常是燃油）
            confidence = min(0.9, 0.5 + yellow_ratio * 0.5)
            return "yellow", "fuel", confidence
        
        return "unknown", "unknown", 0.3
    
    def _classify_rgb(
        self,
        plate_img: np.ndarray
    ) -> Tuple[str, str, float]:
        """基于RGB比值分类（备用方法）
        
        Args:
            plate_img: 车牌图像 (BGR格式)
            
        Returns:
            Tuple[str, str, float]: (颜色, 动力类型, 置信度)
        """
        # 提取底色区域
        h, w = plate_img.shape[:2]
        roi = plate_img[int(h*0.2):int(h*0.8), int(w*0.1):int(w*0.9)]
        
        if roi.size == 0:
            return "unknown", "unknown", 0.0
        
        # 计算平均BGR值
        mean_b = np.mean(roi[:, :, 0])
        mean_g = np.mean(roi[:, :, 1])
        mean_r = np.mean(roi[:, :, 2])
        
        # 避免除零
        total = mean_b + mean_g + mean_r
        if total < 30:
            return "unknown", "unknown", 0.2
        
        # 归一化
        b_norm = mean_b / total
        g_norm = mean_g / total
        r_norm = mean_r / total
        
        # 分类逻辑
        # 电动车: G > B > R (绿色主导)
        # 燃油车: B > G > R (蓝色主导)
        
        if g_norm > b_norm and g_norm > r_norm:
            # 绿色主导
            confidence = min(0.95, 0.5 + (g_norm - b_norm) * 2)
            return "green", "electric", confidence
        elif b_norm > g_norm and b_norm > r_norm:
            # 蓝色主导
            confidence = min(0.95, 0.5 + (b_norm - g_norm) * 2)
            return "blue", "fuel", confidence
        elif r_norm > 0.35 and g_norm > 0.3:
            # 黄色特征 (R和G都较高)
            confidence = min(0.85, 0.4 + (r_norm + g_norm - b_norm))
            return "yellow", "fuel", confidence
        
        return "unknown", "unknown", 0.3
    
    def classify_batch(
        self,
        plate_images: list
    ) -> list:
        """批量分类车牌
        
        Args:
            plate_images: 车牌图像列表
            
        Returns:
            list: PlateClassificationResult列表
        """
        results = []
        for plate_img in plate_images:
            result = self.classify(plate_img)
            results.append(result)
        return results
    
    @property
    def average_processing_time(self) -> float:
        """平均处理时间（毫秒）"""
        if self.classification_count == 0:
            return 0.0
        return self.total_processing_time / self.classification_count
    
    def reset_stats(self) -> None:
        """重置性能统计"""
        self.classification_count = 0
        self.total_processing_time = 0.0
    
    def get_stats(self) -> Dict:
        """获取性能统计信息
        
        Returns:
            Dict: 统计信息字典
        """
        return {
            "classification_count": self.classification_count,
            "average_processing_time_ms": self.average_processing_time,
            "total_processing_time_ms": self.total_processing_time,
            "method": self.method
        }


class VehicleDetectorWithPlate:
    """集成车牌识别的车辆检测器
    
    扩展VehicleDetector，添加车牌颜色识别功能。
    
    Attributes:
        detector: 基础车辆检测器
        plate_classifier: 车牌分类器
        
    Example:
        >>> detector = VehicleDetectorWithPlate('yolo12n.pt')
        >>> detections = detector.detect(frame)
        >>> for det in detections:
        ...     if det.vehicle_power_type == "electric":
        ...         print("电动车 detected")
    """
    
    def __init__(
        self,
        model_path: str = "yolo12n.pt",
        enable_plate: bool = True,
        plate_method: str = "hsv",
        conf_threshold: float = 0.5,
        device: str = "auto"
    ):
        """初始化集成检测器
        
        Args:
            model_path: YOLO模型路径
            enable_plate: 是否启用车牌识别
            plate_method: 车牌分类方法
            conf_threshold: 检测置信度阈值
            device: 推理设备
        """
        # 延迟导入以避免循环依赖
        from .vehicle_detector import VehicleDetector
        
        self.detector = VehicleDetector(
            model_path=model_path,
            conf_threshold=conf_threshold,
            device=device
        )
        
        self.plate_classifier = None
        if enable_plate:
            self.plate_classifier = PlateClassifier(method=plate_method)
        
        self.enable_plate = enable_plate
        
        logger.info(f"集成检测器初始化完成，车牌识别: {enable_plate}")
    
    def detect(
        self,
        frame: np.ndarray,
        detect_plates: bool = True
    ) -> list:
        """检测车辆并识别车牌
        
        Args:
            frame: 输入图像
            detect_plates: 是否检测车牌
            
        Returns:
            list: 检测结果列表（包含车牌信息）
        """
        # 基础车辆检测
        detections = self.detector.detect(frame)
        
        # 车牌识别
        if self.enable_plate and detect_plates and self.plate_classifier:
            for det in detections:
                # 从车辆边界框裁剪车辆图像
                x1, y1, x2, y2 = map(int, det.bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                if x2 > x1 and y2 > y1:
                    vehicle_img = frame[y1:y2, x1:x2]
                    
                    # 检测车牌区域
                    plate_img = self.plate_classifier.detect_plate_region(
                        vehicle_img, det.bbox
                    )
                    
                    if plate_img is not None:
                        # 分类车牌
                        result = self.plate_classifier.classify(plate_img)
                        
                        # 将结果附加到检测对象
                        det.plate_bbox = np.array([0, 0, plate_img.shape[1], plate_img.shape[0]])
                        det.plate_color = result.color
                        det.vehicle_power_type = result.power_type
                        det.plate_confidence = result.confidence
        
        return detections
    
    def detect_and_track(
        self,
        frame: np.ndarray,
        persist: bool = True,
        detect_plates: bool = True
    ) -> list:
        """检测并跟踪车辆（含车牌识别）
        
        Args:
            frame: 输入图像
            persist: 是否保持跟踪器状态
            detect_plates: 是否检测车牌
            
        Returns:
            list: 带track_id的检测结果
        """
        # 基础检测和跟踪
        detections = self.detector.detect_and_track(frame, persist)
        
        # 车牌识别
        if self.enable_plate and detect_plates and self.plate_classifier:
            for det in detections:
                x1, y1, x2, y2 = map(int, det.bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                if x2 > x1 and y2 > y1:
                    vehicle_img = frame[y1:y2, x1:x2]
                    
                    plate_img = self.plate_classifier.detect_plate_region(
                        vehicle_img, det.bbox
                    )
                    
                    if plate_img is not None:
                        result = self.plate_classifier.classify(plate_img)
                        
                        det.plate_bbox = np.array([0, 0, plate_img.shape[1], plate_img.shape[0]])
                        det.plate_color = result.color
                        det.vehicle_power_type = result.power_type
                        det.plate_confidence = result.confidence
        
        return detections
