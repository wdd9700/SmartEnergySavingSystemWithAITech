#!/usr/bin/env python3
"""车辆检测与跟踪模块"""

from .vehicle_detector import VehicleDetector, Detection
from .vehicle_tracker import VehicleTracker, Track
from .speed_estimator import SpeedEstimator
from .camera_processor import CameraProcessor
from .plate_classifier import (
    PlateClassifier,
    PlateClassificationResult,
    PlateColor,
    PowerType,
    VehicleDetectorWithPlate
)

__all__ = [
    'VehicleDetector',
    'Detection', 
    'VehicleTracker',
    'Track',
    'SpeedEstimator',
    'CameraProcessor',
    'PlateClassifier',
    'PlateClassificationResult',
    'PlateColor',
    'PowerType',
    'VehicleDetectorWithPlate'
]
