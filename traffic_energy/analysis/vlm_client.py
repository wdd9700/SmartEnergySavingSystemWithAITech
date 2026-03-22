"""
VLM客户端模块 - 支持Qwen-VL和GPT-4V的视觉语言模型调用
"""

import base64
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class VLMProvider(Enum):
    """VLM提供商枚举"""
    QWEN = "qwen"
    OPENAI = "openai"


@dataclass
class CongestionAnalysisResult:
    """拥堵分析结果"""
    cause: str  # accident | construction | police_control | weather | large_vehicle | signal_failure | unknown
    confidence: float  # 0.0-1.0
    description: str  # 详细描述
    recommended_action: str  # 建议措施
    raw_response: Optional[str] = None  # 原始响应


class VLMClient:
    """
    视觉语言模型客户端
    
    支持多种VLM提供商:
    - Qwen-VL (阿里云)
    - GPT-4V (OpenAI)
    
    提供降级方案: 当VLM不可用时，返回unknown结果
    """
    
    # 拥堵分析提示词模板
    CONGESTION_ANALYSIS_PROMPT = """分析以下交通监控图像，识别造成拥堵的可能原因。

交通数据:
- 车辆密度: {density:.2f} 辆/米
- 平均速度: {avg_speed:.2f} km/h
- 车辆类型分布: {vehicle_types}
- 拥堵持续时间: {duration} 分钟

请分析:
1. 是否可见事故车辆或碎片?
2. 是否有道路施工标志或设备?
3. 是否有交警指挥或临时管制?
4. 是否有特殊天气影响 (积水、大雾等)?
5. 是否有大型车辆阻碍?
6. 其他可能原因?

以JSON格式返回:
{{
    "cause": "accident" | "construction" | "police_control" | "weather" | "large_vehicle" | "signal_failure" | "unknown",
    "confidence": 0.0-1.0,
    "description": "详细描述",
    "recommended_action": "建议措施"
}}"""

    # 有效的拥堵原因
    VALID_CAUSES = [
        "accident",
        "construction", 
        "police_control",
        "weather",
        "large_vehicle",
        "signal_failure",
        "unknown"
    ]
    
    def __init__(
        self,
        provider: Union[str, VLMProvider] = VLMProvider.QWEN,
        api_key: Optional[str] = None,
        fallback_enabled: bool = True,
        confidence_threshold: float = 0.6
    ):
        """
        初始化VLM客户端
        
        Args:
            provider: VLM提供商 ("qwen" | "openai")
            api_key: API密钥，如未提供则从环境变量读取
            fallback_enabled: 是否启用降级方案
            confidence_threshold: 置信度阈值，低于此值标记为unknown
        """
        if isinstance(provider, str):
            provider = VLMProvider(provider.lower())
        self.provider = provider
        self.api_key = api_key or self._get_api_key_from_env()
        self.fallback_enabled = fallback_enabled
        self.confidence_threshold = confidence_threshold
        
        # 初始化API客户端
        self._client = None
        self._init_client()
        
        logger.info(f"VLM客户端初始化完成: provider={provider.value}")
    
    def _get_api_key_from_env(self) -> Optional[str]:
        """从环境变量获取API密钥"""
        if self.provider == VLMProvider.QWEN:
            return os.getenv("DASHSCOPE_API_KEY")
        elif self.provider == VLMProvider.OPENAI:
            return os.getenv("OPENAI_API_KEY")
        return None
    
    def _init_client(self):
        """初始化API客户端"""
        try:
            if self.provider == VLMProvider.QWEN:
                import dashscope
                dashscope.api_key = self.api_key
                self._client = dashscope
            elif self.provider == VLMProvider.OPENAI:
                import openai
                self._client = openai.OpenAI(api_key=self.api_key)
        except ImportError as e:
            logger.warning(f"无法导入VLM库: {e}")
            self._client = None
        except Exception as e:
            logger.warning(f"初始化VLM客户端失败: {e}")
            self._client = None
    
    def _encode_image(self, image: np.ndarray) -> str:
        """
        将numpy图像编码为base64字符串
        
        Args:
            image: numpy数组格式的图像
            
        Returns:
            base64编码的图像字符串
        """
        # 转换为PIL Image
        if len(image.shape) == 2:
            # 灰度图
            pil_image = Image.fromarray(image.astype(np.uint8), mode='L')
        else:
            # 彩色图
            pil_image = Image.fromarray(image.astype(np.uint8))
        
        # 检查图像质量
        if not self._check_image_quality(pil_image):
            logger.warning("图像质量过低，可能影响分析准确性")
        
        # 编码为base64
        import io
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=85)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def _check_image_quality(self, image: Image.Image) -> bool:
        """
        检查图像质量
        
        Args:
            image: PIL Image对象
            
        Returns:
            图像是否可用
        """
        # 检查最小尺寸
        min_size = 224  # 大多数VLM的最小输入尺寸
        if image.width < min_size or image.height < min_size:
            return False
        
        # 检查是否过于模糊（通过简单的方差检查）
        import numpy as np
        img_array = np.array(image.convert('L'))
        variance = np.var(img_array)
        if variance < 100:  # 阈值可根据实际情况调整
            return False
        
        return True
    
    def analyze_congestion(
        self,
        image: np.ndarray,
        traffic_data: Dict
    ) -> CongestionAnalysisResult:
        """
        分析拥堵图像
        
        Args:
            image: 拥堵区域图像 (numpy数组)
            traffic_data: 交通统计数据，包含:
                - density: 车辆密度 (辆/米)
                - avg_speed: 平均速度 (km/h)
                - vehicle_types: 车辆类型分布 (Dict[str, int])
                - duration: 拥堵持续时间 (分钟)
        
        Returns:
            CongestionAnalysisResult: 分析结果
        """
        try:
            # 构建提示词
            prompt = self.CONGESTION_ANALYSIS_PROMPT.format(**traffic_data)
            
            # 调用VLM API
            if self.provider == VLMProvider.QWEN:
                result = self._call_qwen_vl(image, prompt)
            elif self.provider == VLMProvider.OPENAI:
                result = self._call_gpt4v(image, prompt)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")
            
            # 验证和规范化结果
            return self._validate_result(result)
            
        except Exception as e:
            logger.error(f"VLM分析失败: {e}")
            if self.fallback_enabled:
                logger.info("使用降级方案")
                return self._fallback_result(str(e))
            raise
    
    def _call_qwen_vl(
        self,
        image: np.ndarray,
        prompt: str
    ) -> Dict:
        """
        调用Qwen-VL API
        
        Args:
            image: numpy数组格式的图像
            prompt: 提示词
            
        Returns:
            解析后的JSON结果
        """
        if self._client is None:
            raise RuntimeError("Qwen-VL客户端未初始化")
        
        # 编码图像
        image_base64 = self._encode_image(image)
        
        # 构建消息
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    }
                ]
            }
        ]
        
        # 调用API
        response = self._client.MultiModalConversation.call(
            model="qwen-vl-max",
            messages=messages
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Qwen-VL API错误: {response.message}")
        
        # 解析响应
        content = response.output.choices[0].message.content
        return self._parse_vlm_response(content)
    
    def _call_gpt4v(
        self,
        image: np.ndarray,
        prompt: str
    ) -> Dict:
        """
        调用GPT-4V API
        
        Args:
            image: numpy数组格式的图像
            prompt: 提示词
            
        Returns:
            解析后的JSON结果
        """
        if self._client is None:
            raise RuntimeError("OpenAI客户端未初始化")
        
        # 编码图像
        image_base64 = self._encode_image(image)
        
        # 构建消息
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    }
                ]
            }
        ]
        
        # 调用API
        response = self._client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=messages,
            max_tokens=500,
            temperature=0.3
        )
        
        # 解析响应
        content = response.choices[0].message.content
        return self._parse_vlm_response(content)
    
    def _parse_vlm_response(self, content: str) -> Dict:
        """
        解析VLM响应内容
        
        Args:
            content: VLM返回的文本内容
            
        Returns:
            解析后的字典
        """
        try:
            # 尝试直接解析JSON
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # 尝试从文本中提取JSON
        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # 如果无法解析，返回原始内容
        return {
            "cause": "unknown",
            "confidence": 0.0,
            "description": f"无法解析VLM响应: {content[:200]}",
            "recommended_action": "请人工检查图像"
        }
    
    def _validate_result(self, result: Dict) -> CongestionAnalysisResult:
        """
        验证和规范化分析结果
        
        Args:
            result: 原始结果字典
            
        Returns:
            验证后的CongestionAnalysisResult
        """
        # 提取字段
        cause = result.get("cause", "unknown").lower()
        confidence = float(result.get("confidence", 0.0))
        description = result.get("description", "无描述")
        recommended_action = result.get("recommended_action", "请人工检查")
        
        # 验证cause字段
        if cause not in self.VALID_CAUSES:
            logger.warning(f"未知的拥堵原因: {cause}，设置为unknown")
            cause = "unknown"
        
        # 验证confidence范围
        confidence = max(0.0, min(1.0, confidence))
        
        # 应用置信度阈值
        if confidence < self.confidence_threshold:
            logger.info(f"置信度 {confidence:.2f} 低于阈值 {self.confidence_threshold}，标记为unknown")
            cause = "unknown"
        
        return CongestionAnalysisResult(
            cause=cause,
            confidence=confidence,
            description=description,
            recommended_action=recommended_action,
            raw_response=json.dumps(result)
        )
    
    def _fallback_result(self, error_msg: str) -> CongestionAnalysisResult:
        """
        降级方案结果
        
        Args:
            error_msg: 错误信息
            
        Returns:
            默认的unknown结果
        """
        return CongestionAnalysisResult(
            cause="unknown",
            confidence=0.0,
            description=f"VLM分析失败: {error_msg}",
            recommended_action="请人工检查图像或稍后重试"
        )
    
    def health_check(self) -> Dict:
        """
        健康检查
        
        Returns:
            健康状态字典
        """
        status = {
            "provider": self.provider.value,
            "client_initialized": self._client is not None,
            "api_key_configured": self.api_key is not None,
            "fallback_enabled": self.fallback_enabled,
            "confidence_threshold": self.confidence_threshold
        }
        
        # 尝试简单调用测试
        if self._client is not None:
            try:
                # 创建一个简单的测试图像
                test_image = np.zeros((224, 224, 3), dtype=np.uint8)
                # 这里不实际调用API，只检查客户端状态
                status["status"] = "ready"
            except Exception as e:
                status["status"] = "error"
                status["error"] = str(e)
        else:
            status["status"] = "not_initialized"
        
        return status
