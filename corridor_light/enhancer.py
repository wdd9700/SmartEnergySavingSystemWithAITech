#!/usr/bin/env python3
"""
低光照视频增强模块
支持多种增强算法，无需深度学习模型
"""
import cv2
import numpy as np
from typing import Literal


class LowLightEnhancer:
    """低光照图像增强器"""
    
    def __init__(self, method: Literal['clahe', 'gamma', 'msrcr'] = 'clahe',
                 brightness_threshold: int = 50,
                 clip_limit: float = 3.0,
                 tile_size: int = 8,
                 gamma: float = 0.5):
        """
        Args:
            method: 增强算法
            brightness_threshold: 亮度阈值，低于此值才增强
            clip_limit: CLAHE剪切限制
            tile_size: CLAHE网格大小
            gamma: Gamma校正值
        """
        self.method = method
        self.brightness_threshold = brightness_threshold
        self.clip_limit = clip_limit
        self.tile_size = tile_size
        self.gamma = gamma
        
        # 初始化CLAHE
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(tile_size, tile_size)
        )
    
    def estimate_brightness(self, image: np.ndarray) -> float:
        """估计图像亮度（0-255）"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)
    
    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        增强低光照图像
        
        Args:
            image: BGR格式图像
        
        Returns:
            增强后的图像
        """
        if self.method == 'clahe':
            return self._clahe_enhance(image)
        elif self.method == 'gamma':
            return self._gamma_enhance(image)
        elif self.method == 'msrcr':
            return self._msrcr_enhance(image)
        else:
            return image
    
    def _clahe_enhance(self, image: np.ndarray) -> np.ndarray:
        """
        CLAHE增强（对比度受限自适应直方图均衡化）
        最适合夜间低光照场景
        """
        # 转换到LAB色彩空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 对L通道应用CLAHE
        l_enhanced = self.clahe.apply(l)
        
        # 合并通道
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        
        # 转回BGR
        result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # 轻微降噪
        result = cv2.fastNlMeansDenoisingColored(result, None, 3, 3, 7, 21)
        
        return result
    
    def _gamma_enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Gamma校正增强
        简单快速，适合轻微低光照
        """
        # 构建查找表
        inv_gamma = 1.0 / self.gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in range(256)
        ]).astype(np.uint8)
        
        # 应用查找表
        result = cv2.LUT(image, table)
        
        return result
    
    def _msrcr_enhance(self, image: np.ndarray) -> np.ndarray:
        """
        MSRCR增强（多尺度Retinex + 色彩恢复）
        效果最好但计算量稍大
        """
        # 归一化
        img = image.astype(np.float32) / 255.0
        
        # 转换为对数域
        img_log = np.log1p(img)
        
        # 多尺度高斯模糊
        scales = [15, 80, 250]
        retinex = np.zeros_like(img)
        
        for scale in scales:
            blur = cv2.GaussianBlur(img, (0, 0), scale)
            blur_log = np.log1p(blur)
            retinex += img_log - blur_log
        
        retinex /= len(scales)
        
        # 色彩恢复
        img_sum = np.sum(img, axis=2, keepdims=True)
        color_restoration = np.log1p(125 * img / (img_sum + 1e-6))
        
        # 合成
        result = retinex * color_restoration
        
        # 增益和偏移
        gain = 128
        bias = 0
        result = result * gain + bias
        
        # 裁剪到0-255
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        
        return result
    
    def auto_enhance(self, image: np.ndarray) -> tuple:
        """
        自动判断并增强
        
        Returns:
            (增强后的图像, 是否进行了增强)
        """
        brightness = self.estimate_brightness(image)
        
        if brightness < self.brightness_threshold:
            return self.enhance(image), True
        else:
            return image, False
