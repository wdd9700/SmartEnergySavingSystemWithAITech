#!/usr/bin/env python3
"""
人流统计模块
基于YOLOv8，支持跨线计数和密度估计
"""
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import List, Dict, Tuple


class PeopleCounter:
    """人流统计器"""
    
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
        self.input_shape = (640, 640)
        
        # 跟踪相关
        self.trackers = []
        self.next_id = 0
        
    def load_model(self) -> bool:
        """加载模型"""
        if not Path(self.model_path).exists():
            print(f"错误: 模型不存在 {self.model_path}")
            return False
        
        try:
            self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
            input_shape = self.session.get_inputs()[0].shape
            self.input_shape = (input_shape[2], input_shape[3])
            print(f"人流计数模型加载成功")
            return True
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False
    
    def preprocess(self, image: np.ndarray) -> tuple:
        """预处理"""
        h, w = image.shape[:2]
        target_h, target_w = self.input_shape
        
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        
        canvas = canvas.transpose(2, 0, 1).astype(np.float32) / 255.0
        canvas = np.expand_dims(canvas, axis=0)
        
        return canvas, scale, pad_x, pad_y
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """检测人形"""
        if self.session is None:
            return []
        
        orig_h, orig_w = image.shape[:2]
        input_tensor, scale, pad_x, pad_y = self.preprocess(image)
        
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: input_tensor})
        predictions = outputs[0][0].T
        
        scores = predictions[:, 4:].max(axis=1)
        mask = scores > self.conf_threshold
        predictions = predictions[mask]
        
        if len(predictions) == 0:
            return []
        
        boxes = predictions[:, :4]
        class_ids = predictions[:, 4:].argmax(axis=1)
        class_scores = predictions[:, 4:].max(axis=1)
        
        # 转xyxy格式
        boxes_xyxy = boxes.copy()
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        
        # 移除padding并缩放回原始尺寸
        boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad_x) / scale
        boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad_y) / scale
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_w)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_h)
        
        # NMS
        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(), class_scores.tolist(),
            self.conf_threshold, self.iou_threshold
        )
        
        results = []
        if len(indices) > 0:
            for idx in indices.flatten():
                x1, y1, x2, y2 = boxes_xyxy[idx].astype(int)
                results.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(class_scores[idx]),
                    'class': self.CLASSES[class_ids[idx]],
                    'class_id': int(class_ids[idx]),
                    'center': ((x1 + x2) // 2, (y1 + y2) // 2)
                })
        
        return results
    
    def estimate_density(self, detections: List[Dict], frame_shape: tuple) -> float:
        """
        估计人群密度
        
        Returns:
            0-1之间的密度值
        """
        people = [d for d in detections if d['class'] == 'person']
        if not people:
            return 0.0
        
        frame_area = frame_shape[0] * frame_shape[1]
        
        # 计算人的总占用面积
        total_person_area = 0
        for p in people:
            x1, y1, x2, y2 = p['bbox']
            area = (x2 - x1) * (y2 - y1)
            total_person_area += area
        
        # 密度 = 人占用面积 / 总面积
        density = total_person_area / frame_area
        
        return min(density * 5, 1.0)  # 放大系数，封顶1.0
