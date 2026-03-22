"""
偏差分析器测试

测试DeviationAnalyzer的各种功能，包括:
- 偏差度量计算
- 显著偏差检测
- 历史拟合度评估
- 趋势分析
"""

import unittest
from datetime import datetime, timedelta
import numpy as np

# 使用绝对导入避免触发building_energy的__init__.py
import sys
import os

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

# 直接导入模块文件
import importlib.util

def load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# 加载模块
fd_path = os.path.join(project_root, 'building_energy', 'fault_detection')
predictor_module = load_module_from_path('predictor_monitor', os.path.join(fd_path, 'predictor_monitor.py'))
analyzer_module = load_module_from_path('deviation_analyzer', os.path.join(fd_path, 'deviation_analyzer.py'))

PredictionActualPair = predictor_module.PredictionActualPair
DeviationAnalyzer = analyzer_module.DeviationAnalyzer
DeviationMetrics = analyzer_module.DeviationMetrics


class TestDeviationAnalyzer(unittest.TestCase):
    """偏差分析器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.analyzer = DeviationAnalyzer(
            window_size=24,
            temp_threshold=2.0,
            humidity_threshold=10.0,
            power_threshold=1.0,
            historical_fit_threshold=0.7
        )
        
        # 创建测试数据
        self.test_data = self._create_normal_data(n_samples=48)
    
    def _create_normal_data(self, n_samples: int = 24) -> list:
        """创建正常运行的测试数据"""
        data = []
        base_time = datetime.now() - timedelta(hours=n_samples)
        
        for i in range(n_samples):
            # 模拟正常偏差（小偏差）
            temp_noise = np.random.normal(0, 0.3)
            humidity_noise = np.random.normal(0, 2)
            power_noise = np.random.normal(0, 0.1)
            
            pair = PredictionActualPair(
                timestamp=base_time + timedelta(hours=i),
                predicted_temp=24.0,
                actual_temp=24.0 + temp_noise,
                predicted_humidity=50.0,
                actual_humidity=50.0 + humidity_noise,
                predicted_power=2.5,
                actual_power=2.5 + power_noise,
                zone_id="zone_1",
                outdoor_temp=30.0,
                occupancy=5
            )
            data.append(pair)
        
        return data
    
    def _create_faulty_data(self, n_samples: int = 24, fault_type: str = "temperature") -> list:
        """创建带故障的测试数据"""
        data = []
        base_time = datetime.now() - timedelta(hours=n_samples)
        
        for i in range(n_samples):
            if fault_type == "temperature":
                # 模拟温度传感器故障（大偏差）
                temp_noise = np.random.normal(3.0, 0.5)  # 3度偏差
                humidity_noise = np.random.normal(0, 2)
                power_noise = np.random.normal(0, 0.1)
            elif fault_type == "humidity":
                temp_noise = np.random.normal(0, 0.3)
                humidity_noise = np.random.normal(15, 3)  # 15%湿度偏差
                power_noise = np.random.normal(0, 0.1)
            elif fault_type == "power":
                temp_noise = np.random.normal(0, 0.3)
                humidity_noise = np.random.normal(0, 2)
                power_noise = np.random.normal(-1.5, 0.2)  # 功耗异常低
            else:
                temp_noise = np.random.normal(0, 0.3)
                humidity_noise = np.random.normal(0, 2)
                power_noise = np.random.normal(0, 0.1)
            
            pair = PredictionActualPair(
                timestamp=base_time + timedelta(hours=i),
                predicted_temp=24.0,
                actual_temp=24.0 + temp_noise,
                predicted_humidity=50.0,
                actual_humidity=50.0 + humidity_noise,
                predicted_power=2.5,
                actual_power=2.5 + power_noise,
                zone_id="zone_1",
                outdoor_temp=30.0,
                occupancy=5
            )
            data.append(pair)
        
        return data
    
    def test_calculate_metrics_normal(self):
        """测试正常数据的度量计算"""
        metrics = self.analyzer.calculate_metrics(self.test_data)
        
        # 验证返回类型
        self.assertIsInstance(metrics, DeviationMetrics)
        
        # 正常数据的MAE应该较小
        self.assertLess(metrics.temp_mae, 1.0)
        self.assertLess(metrics.humidity_mae, 5.0)
        self.assertLess(metrics.power_mae, 0.5)
        
        # 验证样本数（受window_size限制）
        self.assertEqual(metrics.sample_count, 24)  # window_size=24
    
    def test_calculate_metrics_with_fault(self):
        """测试故障数据的度量计算"""
        faulty_data = self._create_faulty_data(n_samples=24, fault_type="temperature")
        metrics = self.analyzer.calculate_metrics(faulty_data)
        
        # 故障数据的MAE应该较大
        self.assertGreater(metrics.temp_mae, 2.0)
        
        # 验证最大偏差
        self.assertGreater(metrics.temp_max_dev, 2.0)
    
    def test_is_deviation_significant_normal(self):
        """测试正常数据的显著性判断"""
        metrics = self.analyzer.calculate_metrics(self.test_data)
        is_significant = self.analyzer.is_deviation_significant(metrics)
        
        # 正常数据不应该被判定为显著偏差
        self.assertFalse(is_significant)
    
    def test_is_deviation_significant_fault(self):
        """测试故障数据的显著性判断"""
        faulty_data = self._create_faulty_data(n_samples=24, fault_type="temperature")
        metrics = self.analyzer.calculate_metrics(faulty_data)
        is_significant = self.analyzer.is_deviation_significant(metrics)
        
        # 故障数据应该被判定为显著偏差
        self.assertTrue(is_significant)
    
    def test_assess_historical_fit(self):
        """测试历史拟合度评估"""
        # 创建足够的历史数据
        historical_data = self._create_normal_data(n_samples=200)
        
        fit_score = self.analyzer.assess_historical_fit(historical_data)
        
        # 拟合度应该在0-1之间
        self.assertGreaterEqual(fit_score, 0.0)
        self.assertLessEqual(fit_score, 1.0)
        
        # 正常数据的拟合度应该较高
        self.assertGreater(fit_score, 0.5)
    
    def test_assess_historical_fit_insufficient_data(self):
        """测试数据不足时的拟合度评估"""
        # 创建少量数据
        small_data = self._create_normal_data(n_samples=10)
        
        fit_score = self.analyzer.assess_historical_fit(small_data)
        
        # 数据不足时返回默认值
        self.assertEqual(fit_score, 0.8)
    
    def test_is_model_reliable(self):
        """测试模型可靠性判断"""
        # 创建足够的历史数据
        historical_data = self._create_normal_data(n_samples=200)
        
        is_reliable = self.analyzer.is_model_reliable(historical_data)
        
        # 正常数据应该判断为可靠
        self.assertTrue(is_reliable)
    
    def test_get_deviation_trend(self):
        """测试偏差趋势分析"""
        # 创建足够的数据
        data = self._create_normal_data(n_samples=50)
        
        trend = self.analyzer.get_deviation_trend(data, window_size=24)
        
        # 验证返回结构
        self.assertIn('trend', trend)
        self.assertIn('temp_trend', trend)
        self.assertIn('description', trend)
    
    def test_get_deviation_trend_insufficient_data(self):
        """测试数据不足时的趋势分析"""
        small_data = self._create_normal_data(n_samples=10)
        
        trend = self.analyzer.get_deviation_trend(small_data, window_size=24)
        
        # 数据不足时返回特定标记
        self.assertEqual(trend['trend'], 'insufficient_data')
    
    def test_zone_filtering(self):
        """测试区域过滤功能"""
        # 创建多区域数据
        multi_zone_data = []
        base_time = datetime.now() - timedelta(hours=24)
        
        for i in range(24):
            for zone_id in ["zone_1", "zone_2"]:
                pair = PredictionActualPair(
                    timestamp=base_time + timedelta(hours=i),
                    predicted_temp=24.0,
                    actual_temp=24.0 + np.random.normal(0, 0.3),
                    predicted_humidity=50.0,
                    actual_humidity=50.0 + np.random.normal(0, 2),
                    predicted_power=2.5,
                    actual_power=2.5 + np.random.normal(0, 0.1),
                    zone_id=zone_id,
                    outdoor_temp=30.0,
                    occupancy=5
                )
                multi_zone_data.append(pair)
        
        # 测试区域过滤
        metrics_zone1 = self.analyzer.calculate_metrics(multi_zone_data, zone_id="zone_1")
        
        # 验证样本数（应该只有zone_1的数据）
        self.assertEqual(metrics_zone1.sample_count, 24)
    
    def test_update_thresholds(self):
        """测试阈值更新功能"""
        # 更新阈值
        self.analyzer.update_thresholds(
            temp_threshold=3.0,
            humidity_threshold=15.0
        )
        
        # 验证更新后的阈值
        self.assertEqual(self.analyzer.temp_threshold, 3.0)
        self.assertEqual(self.analyzer.humidity_threshold, 15.0)
    
    def test_empty_metrics(self):
        """测试空数据的度量计算"""
        empty_data = []
        metrics = self.analyzer.calculate_metrics(empty_data)
        
        # 空数据应该返回全零的度量
        self.assertEqual(metrics.sample_count, 0)
        self.assertEqual(metrics.temp_mae, 0.0)
    
    def test_metrics_to_dict(self):
        """测试度量指标的字典转换"""
        metrics = self.analyzer.calculate_metrics(self.test_data)
        metrics_dict = metrics.to_dict()
        
        # 验证字典结构
        self.assertIn('temperature', metrics_dict)
        self.assertIn('humidity', metrics_dict)
        self.assertIn('power', metrics_dict)
        self.assertIn('overall', metrics_dict)
        
        # 验证温度子结构
        self.assertIn('mae', metrics_dict['temperature'])
        self.assertIn('rmse', metrics_dict['temperature'])
        self.assertIn('max_deviation', metrics_dict['temperature'])


class TestDeviationMetrics(unittest.TestCase):
    """偏差度量数据结构测试"""
    
    def test_metrics_creation(self):
        """测试度量指标创建"""
        metrics = DeviationMetrics(
            temp_mae=1.5,
            temp_rmse=2.0,
            temp_max_dev=3.5,
            temp_mean_dev=0.5,
            temp_std=1.0,
            humidity_mae=5.0,
            humidity_rmse=6.0,
            humidity_max_dev=10.0,
            power_mae=0.5,
            power_rmse=0.7,
            power_max_dev=1.2,
            power_relative_dev=20.0,
            relative_deviation=0.3,
            sample_count=100,
            time_window_hours=24.0
        )
        
        self.assertEqual(metrics.temp_mae, 1.5)
        self.assertEqual(metrics.sample_count, 100)
    
    def test_metrics_to_dict(self):
        """测试度量指标字典转换"""
        metrics = DeviationMetrics(
            temp_mae=1.5,
            temp_rmse=2.0,
            temp_max_dev=3.5,
            temp_mean_dev=0.5,
            temp_std=1.0,
            humidity_mae=5.0,
            humidity_rmse=6.0,
            humidity_max_dev=10.0,
            power_mae=0.5,
            power_rmse=0.7,
            power_max_dev=1.2,
            power_relative_dev=20.0,
            relative_deviation=0.3,
            sample_count=100,
            time_window_hours=24.0
        )
        
        metrics_dict = metrics.to_dict()
        
        # 验证数值被正确舍入
        self.assertEqual(metrics_dict['temperature']['mae'], 1.5)
        self.assertEqual(metrics_dict['overall']['sample_count'], 100)


if __name__ == '__main__':
    unittest.main()
