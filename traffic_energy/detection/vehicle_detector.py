#!/usr/bin/env python3
"""
车辆检测模块

基于Ultralytics YOLO12的车辆检测器封装。
支持车辆类型分类、GPU/CPU自动切换和批量推理。

Example:
    >>> from traffic_energy.detection import VehicleDetector
    >>> detector = VehicleDetector('yolo12n.pt', conf_threshold=0.5)
    >>> detections = detector.detect(frame)
    >>> for det in detections:
    ...     print(f"检测到 {det.class_name}，置信度: {det.confidence:.2f}")
"""

import time
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError as e:
    ULTRALYTICS_AVAILABLE = False
    YOLO = None
    import warnings
    warnings.warn(f"Ultralytics not installed: {e}. VehicleDetector will not be available.")

try:
    from shared.logger import setup_logger
    logger = setup_logger("vehicle_detector")
except ImportError:
    import logging
    logger = logging.getLogger("vehicle_detector")


# COCO数据集车辆类别映射
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle", 
    5: "bus",
    7: "truck"
}


@dataclass
class Detection:
    """检测结果数据类
    
    Attributes:
        bbox: 边界框坐标 [x1, y1, x2, y2]
        confidence: 置信度分数
        class_id: 类别ID
        class_name: 类别名称
        track_id: 跟踪ID（如果有）
    """
    bbox: np.ndarray
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None
    
    @property
    def center(self) -> Tuple[float, float]:
        """计算边界框中心点"""
        return ((self.bbox[0] + self.bbox[2]) / 2,
                (self.bbox[1] + self.bbox[3]) / 2)
    
    @property
    def width(self) -> float:
        """边界框宽度"""
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> float:
        """边界框高度"""
        return self.bbox[3] - self.bbox[1]
    
    @property
    def area(self) -> float:
        """边界框面积"""
        return self.width * self.height


class VehicleDetector:
    """车辆检测器
    
    基于YOLO12的车辆检测封装类，支持多种输入源和推理优化。
    
    Attributes:
        model: YOLO模型实例
        conf_threshold: 置信度阈值
        device: 推理设备
        
    Example:
        >>> detector = VehicleDetector('yolo12n.pt')
        >>> frame = cv2.imread('traffic.jpg')
        >>> detections = detector.detect(frame)
    """
    
    def __init__(
        self,
        model_path: str = "yolo12n.pt",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "auto",
        classes: Optional[List[int]] = None,
        verbose: bool = False
    ) -> None:
        """初始化车辆检测器
        
        Args:
            model_path: 模型路径或名称
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU阈值
            device: 推理设备 ('auto', 'cpu', 'cuda:0', etc.)
            classes: 过滤的类别列表，默认[2,3,5,7]（车辆）
            verbose: 是否输出详细日志
            
        Raises:
            ImportError: ultralytics未安装
            FileNotFoundError: 模型文件不存在
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "ultralytics未安装，请运行: pip install ultralytics>=8.3.0"
            )
        
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes or [2, 3, 5, 7]  # 默认只检测车辆
        self.verbose = verbose
        
        # 自动选择设备
        if device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"初始化车辆检测器，模型: {model_path}，设备: {self.device}")
        
        # 加载模型
        self._load_model(model_path)
        
        # 性能统计
        self.inference_count = 0
        self.total_inference_time = 0.0
        
    def _load_model(self, model_path: str) -> None:
        """加载YOLO模型
        
        Args:
            model_path: 模型路径或名称
            
        Raises:
            FileNotFoundError: 模型文件不存在
        """
        # 检查是否是预训练模型名称
        if not Path(model_path).exists() and not model_path.endswith('.pt'):
            model_path = f"{model_path}.pt"
            
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            logger.info(f"模型加载成功: {model_path}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def detect(
        self,
        frame: np.ndarray,
        classes: Optional[List[int]] = None
    ) -> List[Detection]:
        """检测单帧图像中的车辆
        
        Args:
            frame: 输入图像 (BGR格式)
            classes: 临时覆盖的类别列表
            
        Returns:
            List[Detection]: 检测结果列表
            
        Example:
            >>> frame = cv2.imread('traffic.jpg')
            >>> detections = detector.detect(frame)
            >>> print(f"检测到 {len(detections)} 辆车")
        """
        start_time = time.time()
        
        detect_classes = classes or self.classes
        
        # 执行推理
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=detect_classes,
            verbose=self.verbose,
            device=self.device
        )[0]
        
        # 解析结果
        detections = []
        
        if results.boxes is not None:
            boxes = results.boxes.cpu().numpy()
            
            for box in boxes:
                bbox = box.xyxy[0].astype(np.float32)
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = VEHICLE_CLASSES.get(class_id, f"class_{class_id}")
                
                detection = Detection(
                    bbox=bbox,
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name
                )
                detections.append(detection)
        
        # 更新统计
        inference_time = time.time() - start_time
        self.inference_count += 1
        self.total_inference_time += inference_time
        
        if self.verbose:
            logger.debug(f"检测到 {len(detections)} 个目标，耗时: {inference_time*1000:.1f}ms")
        
        return detections
    
    def detect_batch(
        self,
        frames: List[np.ndarray],
        batch_size: int = 8
    ) -> List[List[Detection]]:
        """批量检测，提高吞吐量
        
        Args:
            frames: 输入图像列表
            batch_size: 批处理大小
            
        Returns:
            List[List[Detection]]: 每帧的检测结果
            
        Example:
            >>> frames = [cv2.imread(f"frame_{i}.jpg") for i in range(10)]
            >>> all_detections = detector.detect_batch(frames, batch_size=4)
        """
        all_detections = []
        
        # 分批处理
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            
            results = self.model(
                batch,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=self.classes,
                verbose=self.verbose,
                device=self.device
            )
            
            # 解析每帧结果
            for result in results:
                detections = []
                
                if result.boxes is not None:
                    boxes = result.boxes.cpu().numpy()
                    
                    for box in boxes:
                        bbox = box.xyxy[0].astype(np.float32)
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = VEHICLE_CLASSES.get(class_id, f"class_{class_id}")
                        
                        detection = Detection(
                            bbox=bbox,
                            confidence=confidence,
                            class_id=class_id,
                            class_name=class_name
                        )
                        detections.append(detection)
                
                all_detections.append(detections)
        
        return all_detections
    
    def detect_and_track(
        self,
        frame: np.ndarray,
        persist: bool = True
    ) -> List[Detection]:
        """检测并跟踪车辆
        
        使用YOLO内置的跟踪功能（ByteTrack/BoT-SORT）
        
        Args:
            frame: 输入图像
            persist: 是否保持跟踪器状态
            
        Returns:
            List[Detection]: 带track_id的检测结果
        """
        results = self.model.track(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            persist=persist,
            verbose=self.verbose,
            device=self.device
        )[0]
        
        detections = []
        
        if results.boxes is not None:
            boxes = results.boxes.cpu().numpy()
            
            for box in boxes:
                bbox = box.xyxy[0].astype(np.float32)
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = VEHICLE_CLASSES.get(class_id, f"class_{class_id}")
                track_id = int(box.id[0]) if box.id is not None else None
                
                detection = Detection(
                    bbox=bbox,
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name,
                    track_id=track_id
                )
                detections.append(detection)
        
        return detections
    
    def export_model(
        self,
        format: str = "engine",
        half: bool = True,
        int8: bool = False
    ) -> str:
        """导出优化模型
        
        Args:
            format: 导出格式 ('engine'=TensorRT, 'openvino', 'onnx')
            half: 使用FP16半精度
            int8: 使用INT8量化
            
        Returns:
            str: 导出文件路径
        """
        logger.info(f"导出模型为 {format} 格式...")
        
        path = self.model.export(
            format=format,
            half=half,
            int8=int8
        )
        
        logger.info(f"模型导出成功: {path}")
        return str(path)
    
    @property
    def average_inference_time(self) -> float:
        """平均推理时间（毫秒）"""
        if self.inference_count == 0:
            return 0.0
        return (self.total_inference_time / self.inference_count) * 1000
    
    @property
    def fps(self) -> float:
        """当前推理FPS"""
        if self.average_inference_time == 0:
            return 0.0
        return 1000.0 / self.average_inference_time
    
    def reset_stats(self) -> None:
        """重置性能统计"""
        self.inference_count = 0
        self.total_inference_time = 0.0
