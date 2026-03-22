#!/usr/bin/env python3
"""
人因照明与预测性开关模块

提供基于昼夜节律的色温调节和基于运动预测的提前开灯策略。

主要组件:
- CircadianRhythm: 昼夜节律照明控制器
- MotionPredictor: 运动预测器
- PredictiveLightingController: 预测性照明控制器
"""

from .circadian_rhythm import CircadianRhythm
from .motion_predictor import MotionPredictor, MotionEvent, ZoneLayout
from .adaptive_controller import PredictiveLightingController

__version__ = "1.0.0"
__all__ = [
    "CircadianRhythm",
    "MotionPredictor",
    "MotionEvent",
    "ZoneLayout",
    "PredictiveLightingController",
]
