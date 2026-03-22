"""
VLM客户端测试
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis.vlm_client import (
    VLMClient,
    VLMProvider,
    CongestionAnalysisResult
)


class TestVLMClient(unittest.TestCase):
    """测试VLM客户端"""
    
    def setUp(self):
        """测试前准备"""
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.traffic_data = {
            "density": 1.5,
            "avg_speed": 15.0,
            "vehicle_types": {"car": 10, "truck": 2},
            "duration": 30
        }
    
    def test_initialization_default(self):
        """测试默认初始化"""
        with patch.dict('os.environ', {'DASHSCOPE_API_KEY': 'test_key'}):
            client = VLMClient()
            self.assertEqual(client.provider, VLMProvider.QWEN)
            self.assertTrue(client.fallback_enabled)
            self.assertEqual(client.confidence_threshold, 0.6)
    
    def test_initialization_with_provider(self):
        """测试指定提供商初始化"""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
            client = VLMClient(provider=VLMProvider.OPENAI)
            self.assertEqual(client.provider, VLMProvider.OPENAI)
    
    def test_initialization_with_string_provider(self):
        """测试字符串提供商初始化"""
        with patch.dict('os.environ', {'DASHSCOPE_API_KEY': 'test_key'}):
            client = VLMClient(provider="qwen")
            self.assertEqual(client.provider, VLMProvider.QWEN)
    
    def test_encode_image(self):
        """测试图像编码"""
        with patch.dict('os.environ', {'DASHSCOPE_API_KEY': 'test_key'}):
            client = VLMClient()
            encoded = client._encode_image(self.test_image)
            self.assertIsInstance(encoded, str)
            self.assertTrue(len(encoded) > 0)
    
    def test_check_image_quality_valid(self):
        """测试图像质量检查 - 有效图像"""
        from PIL import Image
        
        with patch.dict('os.environ', {'DASHSCOPE_API_KEY': 'test_key'}):
            client = VLMClient()
            valid_image = Image.new('RGB', (640, 480), color='red')
            result = client._check_image_quality(valid_image)
            self.assertTrue(result)
    
    def test_check_image_quality_too_small(self):
        """测试图像质量检查 - 图像过小"""
        from PIL import Image
        
        with patch.dict('os.environ', {'DASHSCOPE_API_KEY': 'test_key'}):
            client = VLMClient()
            small_image = Image.new('RGB', (100, 100), color='red')
            result = client._check_image_quality(small_image)
            self.assertFalse(result)
    
    def test_parse_vlm_response_valid_json(self):
        """测试解析有效JSON响应"""
        with patch.dict('os.environ', {'DASHSCOPE_API_KEY': 'test_key'}):
            client = VLMClient()
            response = '{"cause": "accident", "confidence": 0.85, "description": "test", "recommended_action": "dispatch"}'
            result = client._parse_vlm_response(response)
            self.assertEqual(result["cause"], "accident")
            self.assertEqual(result["confidence"], 0.85)
    
    def test_parse_vlm_response_embedded_json(self):
        """测试解析嵌入在文本中的JSON"""
        with patch.dict('os.environ', {'DASHSCOPE_API_KEY': 'test_key'}):
            client = VLMClient()
            response = 'Here is the result: {"cause": "construction", "confidence": 0.75, "description": "test", "recommended_action": "avoid"} Thank you!'
            result = client._parse_vlm_response(response)
            self.assertEqual(result["cause"], "construction")
    
    def test_parse_vlm_response_invalid(self):
        """测试解析无效响应"""
        with patch.dict('os.environ', {'DASHSCOPE_API_KEY': 'test_key'}):
            client = VLMClient()
            response = "This is not JSON"
            result = client._parse_vlm_response(response)
            self.assertEqual(result["cause"], "unknown")
            self.assertEqual(result["confidence"], 0.0)
    
    def test_validate_result_valid(self):
        """测试验证有效结果"""
        with patch.dict('os.environ', {'DASHSCOPE_API_KEY': 'test_key'}):
            client = VLMClient()
            result = {
                "cause": "accident",
                "confidence": 0.85,
                "description": "Vehicle collision",
                "recommended_action": "Dispatch emergency services"
            }
            validated = client._validate_result(result)
            self.assertEqual(validated.cause, "accident")
            self.assertEqual(validated.confidence, 0.85)
    
    def test_validate_result_invalid_cause(self):
        """测试验证无效原因"""
        with patch.dict('os.environ', {'DASHSCOPE_API_KEY': 'test_key'}):
            client = VLMClient()
            result = {
                "cause": "invalid_cause",
                "confidence": 0.85,
                "description": "test",
                "recommended_action": "test"
            }
            validated = client._validate_result(result)
            self.assertEqual(validated.cause, "unknown")
    
    def test_validate_result_low_confidence(self):
        """测试验证低置信度结果"""
        with patch.dict('os.environ', {'DASHSCOPE_API_KEY': 'test_key'}):
            client = VLMClient(confidence_threshold=0.7)
            result = {
                "cause": "accident",
                "confidence": 0.5,
                "description": "test",
                "recommended_action": "test"
            }
            validated = client._validate_result(result)
            self.assertEqual(validated.cause, "unknown")
    
    def test_fallback_result(self):
        """测试降级方案"""
        with patch.dict('os.environ', {'DASHSCOPE_API_KEY': 'test_key'}):
            client = VLMClient()
            result = client._fallback_result("API error")
            self.assertEqual(result.cause, "unknown")
            self.assertEqual(result.confidence, 0.0)
            self.assertIn("API error", result.description)
    
    def test_health_check(self):
        """测试健康检查"""
        with patch.dict('os.environ', {'DASHSCOPE_API_KEY': 'test_key'}):
            client = VLMClient()
            health = client.health_check()
            self.assertIn("provider", health)
            self.assertIn("client_initialized", health)
            self.assertIn("api_key_configured", health)
    
    @patch('analysis.vlm_client.VLMClient._call_qwen_vl')
    def test_analyze_congestion_qwen(self, mock_call):
        """测试Qwen分析"""
        mock_call.return_value = {
            "cause": "accident",
            "confidence": 0.9,
            "description": "Collision detected",
            "recommended_action": "Dispatch"
        }
        
        with patch.dict('os.environ', {'DASHSCOPE_API_KEY': 'test_key'}):
            client = VLMClient(provider=VLMProvider.QWEN)
            # 模拟客户端初始化
            client._client = MagicMock()
            
            result = client.analyze_congestion(self.test_image, self.traffic_data)
            self.assertEqual(result.cause, "accident")
            self.assertEqual(result.confidence, 0.9)
    
    @patch('analysis.vlm_client.VLMClient._call_gpt4v')
    def test_analyze_congestion_openai(self, mock_call):
        """测试OpenAI分析"""
        mock_call.return_value = {
            "cause": "construction",
            "confidence": 0.8,
            "description": "Road work",
            "recommended_action": "Reroute"
        }
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
            client = VLMClient(provider=VLMProvider.OPENAI)
            # 模拟客户端初始化
            client._client = MagicMock()
            
            result = client.analyze_congestion(self.test_image, self.traffic_data)
            self.assertEqual(result.cause, "construction")
            self.assertEqual(result.confidence, 0.8)


class TestCongestionAnalysisResult(unittest.TestCase):
    """测试CongestionAnalysisResult数据类"""
    
    def test_creation(self):
        """测试创建结果对象"""
        result = CongestionAnalysisResult(
            cause="accident",
            confidence=0.85,
            description="Vehicle collision",
            recommended_action="Dispatch emergency services",
            raw_response='{"cause": "accident"}'
        )
        self.assertEqual(result.cause, "accident")
        self.assertEqual(result.confidence, 0.85)
    
    def test_creation_without_raw(self):
        """测试创建结果对象（无原始响应）"""
        result = CongestionAnalysisResult(
            cause="weather",
            confidence=0.7,
            description="Heavy rain",
            recommended_action="Reduce speed"
        )
        self.assertIsNone(result.raw_response)


if __name__ == '__main__':
    unittest.main()
