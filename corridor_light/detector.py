#!/usr/bin/env python3
"""
人形检测模块
基于YOLOv8 ONNX Runtime，轻量快速
"""
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import List, Dict


class PersonDetector:
    """YOLOv8人形检测器"""
    
    # COCO数据集类别
    CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
               'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
               'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 
               'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
               'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
               'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
               'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
               'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
               'toothbrush']
    
    def __init__(self, model_path: str, conf_threshold: float = 0.5, iou_threshold: float = 0.45):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.session = None
        self.input_shape = (640, 640)  # YOLOv8默认输入尺寸
        
    def load_model(self) -> bool:
        """加载ONNX模型"""
        if not Path(self.model_path).exists():
            print(f"错误: 模型文件不存在 {self.model_path}")
            print("请先运行: python models/download_models.py")
            return False
        
        try:
            # 使用CPU执行，轻量快速
            providers = ['CPUExecutionProvider']
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            
            # 获取输入尺寸
            input_shape = self.session.get_inputs()[0].shape
            self.input_shape = (input_shape[2], input_shape[3])  # H, W
            
            print(f"模型加载成功: {self.model_path}")
            print(f"输入尺寸: {self.input_shape}")
            return True
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """预处理图像"""
        # 等比例缩放 + 填充
        h, w = image.shape[:2]
        target_h, target_w = self.input_shape
        
        # 计算缩放比例
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 缩放
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 创建画布并居中放置
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        
        # 归一化: BGR->RGB, 0-255->0-1
        canvas = canvas.transpose(2, 0, 1)  # HWC->CHW
        canvas = canvas.astype(np.float16) / 255.0
        canvas = np.expand_dims(canvas, axis=0)  # 添加batch维度
        
        return canvas, scale, pad_x, pad_y
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        检测图像中的人形

        Returns:
            检测结果列表，每项包含:
            - bbox: [x1, y1, x2, y2]
            - foot_point: (x, y) 脚底中心位置
            - confidence: 置信度
            - class: 类别名称
            - class_id: 类别ID
        """
        if self.session is None:
            return []
        
        orig_h, orig_w = image.shape[:2]
        
        # 预处理
        input_tensor, scale, pad_x, pad_y = self.preprocess(image)
        
        # 推理
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: input_tensor})
        
        # 解析输出 (YOLOv8格式: [batch, 84, 8400])
        predictions = outputs[0][0]  # [84, 8400]
        
        # 转置: [8400, 84]
        predictions = predictions.T
        
        # 筛选置信度
        scores = predictions[:, 4:].max(axis=1)
        mask = scores > self.conf_threshold
        predictions = predictions[mask]
        scores = scores[mask]
        
        if len(predictions) == 0:
            return []
        
        # 解码边界框
        boxes = predictions[:, :4]
        class_ids = predictions[:, 4:].argmax(axis=1)
        class_scores = predictions[:, 4:].max(axis=1)
        
        # 坐标转换回原图
        # xywh -> xyxy
        boxes_xyxy = boxes.copy()
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
        
        # 移除填充并缩放回原始尺寸
        boxes_xyxy[:, [0, 2]] -= pad_x
        boxes_xyxy[:, [1, 3]] -= pad_y
        boxes_xyxy /= scale
        
        # 限制在图像范围内
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_w)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_h)
        
        # NMS
        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(),
            class_scores.tolist(),
            self.conf_threshold,
            self.iou_threshold
        )
        
        results = []
        if len(indices) > 0:
            for idx in indices.flatten():
                x1, y1, x2, y2 = boxes_xyxy[idx].astype(int)
                # 计算脚底位置 (底部中心点)
                foot_x = (x1 + x2) // 2
                foot_y = y2
                results.append({
                    'bbox': [x1, y1, x2, y2],
                    'foot_point': (foot_x, foot_y),
                    'confidence': float(class_scores[idx]),
                    'class': self.CLASSES[class_ids[idx]],
                    'class_id': int(class_ids[idx])
                })
        
        return results
