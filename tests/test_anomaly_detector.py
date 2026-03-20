"""
异常检测模块单元测试

测试AnomalyDetector和HVACAnomalyDetector的功能。
"""

import os
import sys
import unittest
import tempfile
import shutil
from datetime import datetime, timedelta
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from building_energy.models.anomaly_detector import (
    AnomalyDetector,
    HVACAnomalyDetector,
    AnomalyAlert,
    create_detector_from_config
)


class TestAnomalyAlert(unittest.TestCase):
    """测试AnomalyAlert数据类"""
    
    def test_alert_creation(self):
        """测试告警创建"""
        alert = AnomalyAlert(
            timestamp=datetime.now(),
            anomaly_score=0.85,
            anomaly_type='hvac_high_power',
            severity='high',
            description='Test alert',
            affected_metrics=['hvac_power']
        )
        
        self.assertEqual(alert.anomaly_score, 0.85)
        self.assertEqual(alert.severity, 'high')
        self.assertEqual(alert.anomaly_type, 'hvac_high_power')
    
    def test_alert_to_dict(self):
        """测试告警转字典"""
        now = datetime.now()
        alert = AnomalyAlert(
            timestamp=now,
            anomaly_score=0.9,
            anomaly_type='temperature_anomaly',
            severity='critical',
            description='Temperature too high',
            affected_metrics=['indoor_temp']
        )
        
        d = alert.to_dict()
        self.assertEqual(d['anomaly_score'], 0.9)
        self.assertEqual(d['severity'], 'critical')
        self.assertEqual(d['affected_metrics'], ['indoor_temp'])
        self.assertIn('timestamp', d)


class TestAnomalyDetector(unittest.TestCase):
    """测试AnomalyDetector类"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.temp_dir = tempfile.mkdtemp()
        cls.model_path = os.path.join(cls.temp_dir, 'test_model.pkl')
    
    @classmethod
    def tearDownClass(cls):
        """测试类清理"""
        shutil.rmtree(cls.temp_dir, ignore_errors=True)
    
    def setUp(self):
        """每个测试前初始化"""
        # 生成测试数据
        np.random.seed(42)
        self.n_samples = 200
        self.n_features = 5
        
        # 正常数据
        self.normal_data = np.random.randn(self.n_samples, self.n_features)
        
        # 包含异常的数据
        self.mixed_data = np.vstack([
            self.normal_data,
            np.random.randn(20, self.n_features) * 5  # 异常点
        ])
    
    def test_init_default(self):
        """测试默认初始化"""
        detector = AnomalyDetector()
        self.assertEqual(detector.algorithm, 'iforest')
        self.assertEqual(detector.contamination, 0.1)
        self.assertEqual(detector.alert_threshold, 0.8)
        self.assertFalse(detector.is_fitted)
    
    def test_init_custom(self):
        """测试自定义初始化"""
        detector = AnomalyDetector(
            algorithm='lof',
            contamination=0.15,
            alert_threshold=0.7,
            feature_names=['a', 'b', 'c']
        )
        self.assertEqual(detector.algorithm, 'lof')
        self.assertEqual(detector.contamination, 0.15)
        self.assertEqual(detector.alert_threshold, 0.7)
        self.assertEqual(detector.feature_names, ['a', 'b', 'c'])
    
    def test_init_invalid_algorithm(self):
        """测试无效算法"""
        with self.assertRaises(ValueError) as context:
            AnomalyDetector(algorithm='invalid')
        self.assertIn('Unsupported algorithm', str(context.exception))
    
    def test_init_invalid_contamination(self):
        """测试无效contamination"""
        with self.assertRaises(ValueError):
            AnomalyDetector(contamination=0.0)
        
        with self.assertRaises(ValueError):
            AnomalyDetector(contamination=0.6)
    
    def test_fit(self):
        """测试模型训练"""
        detector = AnomalyDetector(algorithm='iforest')
        detector.fit(self.normal_data)
        
        self.assertTrue(detector.is_fitted)
        self.assertIsNotNone(detector.model)
        self.assertEqual(detector.training_stats['n_samples'], self.n_samples)
        self.assertEqual(detector.training_stats['n_features'], self.n_features)
    
    def test_fit_invalid_data(self):
        """测试无效训练数据"""
        detector = AnomalyDetector()
        
        # 1D数据
        with self.assertRaises(ValueError):
            detector.fit(np.array([1, 2, 3]))
        
        # 数据太少
        with self.assertRaises(ValueError):
            detector.fit(np.random.randn(5, 3))
    
    def test_predict(self):
        """测试预测"""
        detector = AnomalyDetector(algorithm='iforest', contamination=0.1)
        detector.fit(self.normal_data)
        
        predictions = detector.predict(self.mixed_data)
        
        self.assertEqual(len(predictions), len(self.mixed_data))
        self.assertTrue(all(p in [0, 1] for p in predictions))
        # 应该检测到一些异常
        self.assertGreater(np.sum(predictions), 0)
    
    def test_predict_not_fitted(self):
        """测试未训练就预测"""
        detector = AnomalyDetector()
        
        with self.assertRaises(RuntimeError) as context:
            detector.predict(self.normal_data)
        self.assertIn('not fitted', str(context.exception).lower())
    
    def test_predict_proba(self):
        """测试概率预测"""
        detector = AnomalyDetector(algorithm='iforest')
        detector.fit(self.normal_data)
        
        proba = detector.predict_proba(self.mixed_data)
        
        self.assertEqual(proba.shape, (len(self.mixed_data), 2))
        # 概率和应该接近1
        np.testing.assert_array_almost_equal(
            np.sum(proba, axis=1),
            np.ones(len(self.mixed_data)),
            decimal=5
        )
        # 概率在0-1之间
        self.assertTrue(np.all(proba >= 0) and np.all(proba <= 1))
    
    def test_decision_function(self):
        """测试异常分数计算"""
        detector = AnomalyDetector(algorithm='iforest')
        detector.fit(self.normal_data)
        
        scores = detector.decision_function(self.mixed_data)
        
        self.assertEqual(len(scores), len(self.mixed_data))
        # 分数应该是数值
        self.assertTrue(np.all(np.isfinite(scores)))
    
    def test_save_load(self):
        """测试模型保存和加载"""
        # 创建并训练模型
        detector1 = AnomalyDetector(
            algorithm='iforest',
            contamination=0.15,
            alert_threshold=0.75,
            feature_names=['f1', 'f2', 'f3', 'f4', 'f5']
        )
        detector1.fit(self.normal_data)
        
        # 保存
        detector1.save(self.model_path)
        self.assertTrue(os.path.exists(self.model_path))
        
        # 加载到新实例
        detector2 = AnomalyDetector()
        detector2.load(self.model_path)
        
        # 验证配置
        self.assertEqual(detector2.algorithm, 'iforest')
        self.assertEqual(detector2.contamination, 0.15)
        self.assertEqual(detector2.alert_threshold, 0.75)
        self.assertEqual(detector2.feature_names, ['f1', 'f2', 'f3', 'f4', 'f5'])
        self.assertTrue(detector2.is_fitted)
        
        # 验证预测结果一致
        pred1 = detector1.predict(self.mixed_data)
        pred2 = detector2.predict(self.mixed_data)
        np.testing.assert_array_equal(pred1, pred2)
    
    def test_load_not_exist(self):
        """测试加载不存在的文件"""
        detector = AnomalyDetector()
        
        with self.assertRaises(FileNotFoundError):
            detector.load('/nonexistent/path/model.pkl')
    
    def test_save_not_fitted(self):
        """测试未训练就保存"""
        detector = AnomalyDetector()
        
        with self.assertRaises(RuntimeError):
            detector.save(self.model_path)
    
    def test_get_alerts(self):
        """测试获取告警"""
        detector = AnomalyDetector(
            algorithm='iforest',
            contamination=0.3,  # 高contamination以产生更多异常
            alert_threshold=0.3
        )
        detector.fit(self.normal_data)
        
        # 触发预测以产生告警
        detector.predict(self.mixed_data)
        
        alerts = detector.get_alerts()
        self.assertIsInstance(alerts, list)
        
        # 测试按严重程度过滤
        high_alerts = detector.get_alerts(severity='high')
        self.assertIsInstance(high_alerts, list)
        
        # 测试按时间过滤
        recent_alerts = detector.get_alerts(since=datetime.now() - timedelta(hours=1))
        self.assertIsInstance(recent_alerts, list)
    
    def test_clear_alerts(self):
        """测试清空告警"""
        detector = AnomalyDetector(
            algorithm='iforest',
            contamination=0.3,
            alert_threshold=0.3
        )
        detector.fit(self.normal_data)
        detector.predict(self.mixed_data)
        
        # 清空告警
        detector.clear_alerts()
        self.assertEqual(len(detector.alert_history), 0)
    
    def test_get_model_info(self):
        """测试获取模型信息"""
        detector = AnomalyDetector(
            algorithm='hbos',
            contamination=0.12,
            feature_names=['a', 'b']
        )
        
        info = detector.get_model_info()
        
        self.assertEqual(info['algorithm'], 'hbos')
        self.assertEqual(info['contamination'], 0.12)
        self.assertEqual(info['feature_names'], ['a', 'b'])
        self.assertFalse(info['is_fitted'])
        self.assertIn('supported_algorithms', info)
    
    def test_different_algorithms(self):
        """测试不同算法"""
        algorithms = ['iforest', 'lof', 'hbos']
        
        for algo in algorithms:
            with self.subTest(algorithm=algo):
                detector = AnomalyDetector(algorithm=algo)
                detector.fit(self.normal_data)
                predictions = detector.predict(self.mixed_data)
                self.assertEqual(len(predictions), len(self.mixed_data))


class TestHVACAnomalyDetector(unittest.TestCase):
    """测试HVACAnomalyDetector类"""
    
    def setUp(self):
        """测试前初始化"""
        np.random.seed(42)
        
        # 生成HVAC测试数据 (8个特征)
        self.n_samples = 150
        self.hvac_data = np.random.randn(self.n_samples, 8)
    
    def test_init(self):
        """测试初始化"""
        detector = HVACAnomalyDetector()
        
        self.assertEqual(detector.algorithm, 'iforest')
        self.assertEqual(detector.feature_names, HVACAnomalyDetector.HVAC_FEATURES)
        self.assertEqual(len(detector.feature_names), 8)
    
    def test_init_custom_thresholds(self):
        """测试自定义阈值"""
        detector = HVACAnomalyDetector(
            temp_threshold=3.0,
            power_threshold=5.0
        )
        
        self.assertEqual(detector.temp_threshold, 3.0)
        self.assertEqual(detector.power_threshold, 5.0)
    
    def test_monitor(self):
        """测试监控功能"""
        detector = HVACAnomalyDetector(algorithm='iforest')
        detector.fit(self.hvac_data)
        
        result = detector.monitor(self.hvac_data)
        
        self.assertIn('timestamp', result)
        self.assertIn('is_anomaly', result)
        self.assertIn('anomaly_probability', result)
        self.assertIn('anomaly_score', result)
        self.assertIn('active_alerts', result)
        self.assertIn('system_status', result)
        
        self.assertIsInstance(result['is_anomaly'], bool)
        self.assertIsInstance(result['anomaly_probability'], float)
        self.assertIsInstance(result['active_alerts'], list)
    
    def test_assess_system_status(self):
        """测试系统状态评估"""
        detector = HVACAnomalyDetector()
        
        # 正常状态
        normal_data = np.zeros(8)
        status = detector._assess_system_status(normal_data)
        self.assertEqual(status, 'normal')
        
        # 高功耗
        high_power_data = np.zeros(8)
        high_power_data[3] = 5.0  # hvac_power索引
        status = detector._assess_system_status(high_power_data)
        self.assertIn('high_power', status)
        
        # 温度偏差
        temp_dev_data = np.zeros(8)
        temp_dev_data[1] = 3.0  # indoor_temp索引
        status = detector._assess_system_status(temp_dev_data)
        self.assertIn('temp_deviation', status)
        
        # 无数据
        status = detector._assess_system_status(None)
        self.assertEqual(status, 'unknown')


class TestCreateDetectorFromConfig(unittest.TestCase):
    """测试从配置创建检测器"""
    
    def test_create_general_detector(self):
        """测试创建通用检测器"""
        config = {
            'type': 'general',
            'algorithm': 'iforest',
            'contamination': 0.1
        }
        
        detector = create_detector_from_config(config)
        
        self.assertIsInstance(detector, AnomalyDetector)
        self.assertEqual(detector.algorithm, 'iforest')
        self.assertEqual(detector.contamination, 0.1)
    
    def test_create_hvac_detector(self):
        """测试创建HVAC检测器"""
        config = {
            'type': 'hvac',
            'algorithm': 'lof',
            'contamination': 0.15,
            'temp_threshold': 2.5
        }
        
        detector = create_detector_from_config(config)
        
        self.assertIsInstance(detector, HVACAnomalyDetector)
        self.assertEqual(detector.algorithm, 'lof')
        self.assertEqual(detector.temp_threshold, 2.5)
    
    def test_create_default_type(self):
        """测试默认类型"""
        config = {
            'algorithm': 'hbos'
        }
        
        detector = create_detector_from_config(config)
        
        self.assertIsInstance(detector, AnomalyDetector)
        self.assertEqual(detector.algorithm, 'hbos')


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_full_workflow(self):
        """测试完整工作流程"""
        np.random.seed(42)
        
        # 1. 创建检测器
        detector = HVACAnomalyDetector(
            algorithm='iforest',
            contamination=0.1,
            alert_threshold=0.7
        )
        
        # 2. 生成训练数据
        train_data = np.random.randn(300, 8)
        
        # 3. 训练模型
        detector.fit(train_data)
        
        # 4. 生成测试数据（包含异常）
        test_data = np.vstack([
            np.random.randn(50, 8),
            np.random.randn(10, 8) * 4  # 异常
        ])
        
        # 5. 预测
        predictions = detector.predict(test_data)
        proba = detector.predict_proba(test_data)
        
        # 6. 监控
        monitor_result = detector.monitor(test_data)
        
        # 7. 验证结果
        self.assertEqual(len(predictions), 60)
        self.assertEqual(proba.shape, (60, 2))
        self.assertIn('system_status', monitor_result)
        
        # 8. 获取告警
        alerts = detector.get_alerts()
        self.assertIsInstance(alerts, list)
        
        # 9. 保存和加载
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name
        
        try:
            detector.save(temp_path)
            
            new_detector = HVACAnomalyDetector()
            new_detector.load(temp_path)
            
            # 验证加载后的模型能正常工作
            new_predictions = new_detector.predict(test_data)
            np.testing.assert_array_equal(predictions, new_predictions)
        finally:
            os.unlink(temp_path)


def run_tests():
    """运行所有测试"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加所有测试类
    suite.addTests(loader.loadTestsFromTestCase(TestAnomalyAlert))
    suite.addTests(loader.loadTestsFromTestCase(TestAnomalyDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestHVACAnomalyDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestCreateDetectorFromConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
