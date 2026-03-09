#!/usr/bin/env python3
"""
使用OpenCV DNN代替ONNX Runtime的备选检测器
当YOLOv8模型不可用时使用
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict


class FallbackDetector:
    """备选检测器 - 使用OpenCV DNN或Haar级联"""
    
    def __init__(self):
        self.detector = None
        self._init_detector()
    
    def _init_detector(self):
        """初始化检测器"""
        # 尝试使用OpenCV内置的HOG行人检测器
        try:
            self.detector = cv2.HOGDescriptor()
            self.detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            print("使用HOG行人检测器")
        except:
            print("警告: 无法初始化检测器")
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """检测人形"""
        if self.detector is None:
            return []
        
        # HOG检测
        rects, weights = self.detector.detectMultiScale(
            image, 
            winStride=(8, 8),
            padding=(4, 4),
            scale=1.05
        )
        
        results = []
        for i, (x, y, w, h) in enumerate(rects):
            results.append({
                'bbox': [int(x), int(y), int(x+w), int(y+h)],
                'confidence': float(weights[i]),
                'class': 'person',
                'class_id': 0
            })
        
        return results


if __name__ == "__main__":
    # 测试
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from tests.create_test_videos import create_test_image_multiple_people
    
    detector = FallbackDetector()
    test_img = create_test_image_multiple_people()
    
    detections = detector.detect(test_img)
    print(f"检测到 {len(detections)} 个人")
    for d in detections:
        print(f"  置信度: {d['confidence']:.2f}, 位置: {d['bbox']}")
