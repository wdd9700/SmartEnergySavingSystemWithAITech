"""
能耗预测模型单元测试

测试EnergyPredictor和BaselineModel的功能。
"""

import unittest
import tempfile
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any

import numpy as np
import pandas as pd

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from building_energy.models.predictor import (
    EnergyPredictor, LSTMModel, GRUModel, MLPModel,
    TimeSeriesDataset, PredictionResult
)
from building_energy.models.baseline import (
    BaselineModel, BaselineType,
    HistoricalAverageBaseline, RegressionBaseline, ClusteringBaseline
)


class TestLSTMModel(unittest.TestCase):
    """测试LSTM模型"""
    
    def setUp(self):
        """测试前准备"""
        self.input_size = 10
        self.hidden_size = 64
        self.num_layers = 2
        self.batch_size = 4
        self.seq_length = 24
        
        self.model = LSTMModel(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=1
        )
    
    def test_model_creation(self):
        """测试模型创建"""
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.hidden_size, self.hidden_size)
        self.assertEqual(self.model.num_layers, self.num_layers)
    
    def test_forward_pass(self):
        """测试前向传播"""
        import torch
        
        x = torch.randn(self.batch_size, self.seq_length, self.input_size)
        output = self.model(x)
        
        self.assertEqual(output.shape, (self.batch_size, 1))


class TestGRUModel(unittest.TestCase):
    """测试GRU模型"""
    
    def setUp(self):
        """测试前准备"""
        self.input_size = 10
        self.hidden_size = 64
        self.batch_size = 4
        self.seq_length = 24
        
        self.model = GRUModel(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=2,
            output_size=1
        )
    
    def test_forward_pass(self):
        """测试前向传播"""
        import torch
        
        x = torch.randn(self.batch_size, self.seq_length, self.input_size)
        output = self.model(x)
        
        self.assertEqual(output.shape, (self.batch_size, 1))


class TestMLPModel(unittest.TestCase):
    """测试MLP模型"""
    
    def setUp(self):
        """测试前准备"""
        self.input_size = 10
        self.seq_length = 24
        self.batch_size = 4
        
        self.model = MLPModel(
            input_size=self.input_size,
            hidden_sizes=[64, 32],
            output_size=1
        )
    
    def test_forward_pass(self):
        """测试前向传播"""
        import torch
        
        x = torch.randn(self.batch_size, self.seq_length, self.input_size)
        output = self.model(x)
        
        self.assertEqual(output.shape, (self.batch_size, 1))


class TestTimeSeriesDataset(unittest.TestCase):
    """测试时间序列数据集"""
    
    def setUp(self):
        """测试前准备"""
        np.random.seed(42)
        self.data = np.random.randn(100, 5)
        self.targets = np.random.randn(100)
        self.seq_length = 24
        
        self.dataset = TimeSeriesDataset(
            data=self.data,
            targets=self.targets,
            seq_length=self.seq_length
        )
    
    def test_dataset_length(self):
        """测试数据集长度"""
        expected_length = len(self.data) - self.seq_length
        self.assertEqual(len(self.dataset), expected_length)
    
    def test_get_item(self):
        """测试获取数据项"""
        import torch
        
        x, y = self.dataset[0]
        
        self.assertIsInstance(x, torch.FloatTensor)
        self.assertIsInstance(y, torch.FloatTensor)
        self.assertEqual(x.shape, (self.seq_length, 5))
        self.assertEqual(y.shape, (1,))


class TestEnergyPredictor(unittest.TestCase):
    """测试能耗预测器"""
    
    def setUp(self):
        """测试前准备"""
        np.random.seed(42)
        
        # 创建测试数据
        n_samples = 200
        timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq='H')
        
        self.test_data = pd.DataFrame({
            'timestamp': timestamps,
            'outdoor_temp': 25 + 5 * np.sin(np.arange(n_samples) * 2 * np.pi / 24) + np.random.randn(n_samples) * 2,
            'indoor_temp': 24 + 2 * np.sin(np.arange(n_samples) * 2 * np.pi / 24) + np.random.randn(n_samples) * 1,
            'indoor_humidity': 50 + 10 * np.sin(np.arange(n_samples) * 2 * np.pi / 24) + np.random.randn(n_samples) * 5,
            'solar_radiation': np.maximum(0, 800 * np.sin(np.arange(n_samples) * np.pi / 12)) + np.random.randn(n_samples) * 50,
            'occupancy': np.random.randint(0, 30, n_samples),
            'hvac_power': 3 + 2 * np.sin(np.arange(n_samples) * 2 * np.pi / 24) + np.random.randn(n_samples) * 0.5
        })
        
        self.predictor = EnergyPredictor(
            model_type="mlp",
            seq_length=12,
            hidden_size=32,
            num_layers=1,
            epochs=10
        )
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.predictor.model_type, "mlp")
        self.assertEqual(self.predictor.seq_length, 12)
        self.assertIsNone(self.predictor.model)
    
    def test_prepare_features(self):
        """测试特征工程"""
        df = self.predictor._prepare_features(self.test_data)
        
        # 检查是否添加了时间特征
        self.assertIn('hour', df.columns)
        self.assertIn('day_of_week', df.columns)
        self.assertIn('is_holiday', df.columns)
        self.assertIn('hour_sin', df.columns)
        self.assertIn('hour_cos', df.columns)
    
    def test_train(self):
        """测试训练"""
        history = self.predictor.train(
            data=self.test_data,
            target_column='hvac_power',
            epochs=5,
            batch_size=16,
            validation_split=0.2,
            verbose=False
        )
        
        # 检查训练历史
        self.assertIn('train_loss', history)
        self.assertIn('val_loss', history)
        self.assertGreater(len(history['train_loss']), 0)
        
        # 检查模型是否已创建
        self.assertIsNotNone(self.predictor.model)
    
    def test_predict(self):
        """测试预测"""
        # 先训练模型
        self.predictor.train(
            data=self.test_data,
            target_column='hvac_power',
            epochs=5,
            batch_size=16,
            verbose=False
        )
        
        # 预测
        result = self.predictor.predict(
            data=self.test_data,
            horizon=12
        )
        
        # 检查结果
        self.assertIsInstance(result, PredictionResult)
        self.assertEqual(len(result.predictions), 12)
        self.assertEqual(len(result.timestamps), 12)
    
    def test_evaluate(self):
        """测试评估"""
        # 先训练模型
        self.predictor.train(
            data=self.test_data[:150],
            target_column='hvac_power',
            epochs=5,
            batch_size=16,
            verbose=False
        )
        
        # 评估
        metrics = self.predictor.evaluate(
            test_data=self.test_data[150:],
            target_column='hvac_power'
        )
        
        # 检查评估指标
        self.assertIn('mae', metrics)
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mape', metrics)
        self.assertIn('r2', metrics)
    
    def test_save_load(self):
        """测试保存和加载"""
        # 训练模型
        self.predictor.train(
            data=self.test_data,
            target_column='hvac_power',
            epochs=5,
            batch_size=16,
            verbose=False
        )
        
        # 保存模型
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            temp_path = f.name
        
        try:
            self.predictor.save(temp_path)
            self.assertTrue(os.path.exists(temp_path))
            
            # 创建新预测器并加载
            new_predictor = EnergyPredictor(model_type="mlp", seq_length=12)
            new_predictor.load(temp_path)
            
            # 检查配置是否恢复
            self.assertEqual(new_predictor.model_type, self.predictor.model_type)
            self.assertEqual(new_predictor.seq_length, self.predictor.seq_length)
            self.assertIsNotNone(new_predictor.model)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestHistoricalAverageBaseline(unittest.TestCase):
    """测试历史平均基线模型"""
    
    def setUp(self):
        """测试前准备"""
        np.random.seed(42)
        
        n_samples = 24 * 14  # 两周数据
        timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq='H')
        
        self.test_data = pd.DataFrame({
            'timestamp': timestamps,
            'hvac_power': 5 + 2 * np.sin(np.arange(n_samples) * 2 * np.pi / 24) + np.random.randn(n_samples) * 0.5
        })
        
        self.model = HistoricalAverageBaseline()
    
    def test_fit(self):
        """测试训练"""
        self.model.fit(self.test_data, target_column='hvac_power')
        
        # 检查是否计算了基线统计
        self.assertGreater(len(self.model.baseline_stats), 0)
    
    def test_predict(self):
        """测试预测"""
        self.model.fit(self.test_data, target_column='hvac_power')
        
        # 预测未来24小时
        future_timestamps = pd.date_range(
            start=self.test_data['timestamp'].iloc[-1] + timedelta(hours=1),
            periods=24,
            freq='H'
        )
        
        predictions = self.model.predict(future_timestamps.tolist())
        
        self.assertEqual(len(predictions), 24)
    
    def test_save_load(self):
        """测试保存和加载"""
        self.model.fit(self.test_data, target_column='hvac_power')
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            self.model.save(temp_path)
            
            new_model = HistoricalAverageBaseline()
            new_model.load(temp_path)
            
            self.assertEqual(
                self.model.baseline_stats,
                new_model.baseline_stats
            )
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestRegressionBaseline(unittest.TestCase):
    """测试回归基线模型"""
    
    def setUp(self):
        """测试前准备"""
        np.random.seed(42)
        
        n_samples = 24 * 7
        timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq='H')
        
        self.test_data = pd.DataFrame({
            'timestamp': timestamps,
            'outdoor_temp': 25 + 5 * np.sin(np.arange(n_samples) * 2 * np.pi / 24),
            'indoor_temp': 24 + 2 * np.sin(np.arange(n_samples) * 2 * np.pi / 24),
            'hvac_power': 5 + 2 * np.sin(np.arange(n_samples) * 2 * np.pi / 24) + np.random.randn(n_samples) * 0.5
        })
        
        self.model = RegressionBaseline(model_type='ridge')
    
    def test_fit(self):
        """测试训练"""
        self.model.fit(self.test_data, target_column='hvac_power')
        
        self.assertTrue(self.model.is_fitted)
        self.assertIsNotNone(self.model.model)
    
    def test_predict(self):
        """测试预测"""
        self.model.fit(self.test_data, target_column='hvac_power')
        
        predictions = self.model.predict(self.test_data)
        
        self.assertEqual(len(predictions), len(self.test_data))
    
    def test_feature_importance(self):
        """测试特征重要性"""
        self.model.fit(self.test_data, target_column='hvac_power')
        
        importance = self.model.get_feature_importance()
        
        self.assertGreater(len(importance), 0)


class TestClusteringBaseline(unittest.TestCase):
    """测试聚类基线模型"""
    
    def setUp(self):
        """测试前准备"""
        np.random.seed(42)
        
        n_samples = 24 * 14
        timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq='H')
        
        self.test_data = pd.DataFrame({
            'timestamp': timestamps,
            'outdoor_temp': 25 + 5 * np.sin(np.arange(n_samples) * 2 * np.pi / 24),
            'hvac_power': 5 + 2 * np.sin(np.arange(n_samples) * 2 * np.pi / 24) + np.random.randn(n_samples) * 0.5
        })
        
        self.model = ClusteringBaseline(n_clusters=3)
    
    def test_fit(self):
        """测试训练"""
        self.model.fit(self.test_data, target_column='hvac_power')
        
        self.assertTrue(self.model.is_fitted)
        self.assertEqual(len(self.model.cluster_profiles), 3)
    
    def test_predict(self):
        """测试预测"""
        self.model.fit(self.test_data, target_column='hvac_power')
        
        predictions = self.model.predict(self.test_data)
        
        self.assertEqual(len(predictions), len(self.test_data))
    
    def test_cluster_info(self):
        """测试聚类信息"""
        self.model.fit(self.test_data, target_column='hvac_power')
        
        info = self.model.get_cluster_info()
        
        self.assertEqual(len(info), 3)
        for cluster_id, stats in info.items():
            self.assertIn('mean', stats)
            self.assertIn('std', stats)


class TestBaselineModel(unittest.TestCase):
    """测试统一基线模型接口"""
    
    def setUp(self):
        """测试前准备"""
        np.random.seed(42)
        
        n_samples = 24 * 14
        timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq='H')
        
        self.test_data = pd.DataFrame({
            'timestamp': timestamps,
            'outdoor_temp': 25 + 5 * np.sin(np.arange(n_samples) * 2 * np.pi / 24),
            'hvac_power': 5 + 2 * np.sin(np.arange(n_samples) * 2 * np.pi / 24) + np.random.randn(n_samples) * 0.5
        })
    
    def test_historical_baseline(self):
        """测试历史平均基线"""
        model = BaselineModel(BaselineType.HISTORICAL_AVERAGE)
        model.fit(self.test_data, target_column='hvac_power')
        
        result = model.evaluate(self.test_data, target_column='hvac_power')
        
        self.assertIsNotNone(result.baseline_values)
        self.assertIsNotNone(result.savings)
        self.assertIsNotNone(result.mape)
    
    def test_regression_baseline(self):
        """测试回归基线"""
        model = BaselineModel(BaselineType.REGRESSION)
        model.fit(self.test_data, target_column='hvac_power')
        
        result = model.evaluate(self.test_data, target_column='hvac_power')
        
        self.assertIsNotNone(result.baseline_values)
        self.assertIsNotNone(result.mape)
    
    def test_clustering_baseline(self):
        """测试聚类基线"""
        model = BaselineModel(BaselineType.CLUSTERING, n_clusters=3)
        model.fit(self.test_data, target_column='hvac_power')
        
        result = model.evaluate(self.test_data, target_column='hvac_power')
        
        self.assertIsNotNone(result.baseline_values)
        self.assertIsNotNone(result.mape)


class TestModelTypes(unittest.TestCase):
    """测试不同模型类型"""
    
    def setUp(self):
        """测试前准备"""
        np.random.seed(42)
        
        n_samples = 200
        timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq='H')
        
        self.test_data = pd.DataFrame({
            'timestamp': timestamps,
            'outdoor_temp': 25 + 5 * np.sin(np.arange(n_samples) * 2 * np.pi / 24),
            'indoor_temp': 24 + 2 * np.sin(np.arange(n_samples) * 2 * np.pi / 24),
            'indoor_humidity': 50 + np.random.randn(n_samples) * 5,
            'solar_radiation': np.maximum(0, 800 * np.sin(np.arange(n_samples) * np.pi / 12)),
            'occupancy': np.random.randint(0, 30, n_samples),
            'hvac_power': 3 + 2 * np.sin(np.arange(n_samples) * 2 * np.pi / 24) + np.random.randn(n_samples) * 0.5
        })
    
    def test_lstm_model(self):
        """测试LSTM模型"""
        predictor = EnergyPredictor(
            model_type="lstm",
            seq_length=12,
            hidden_size=32,
            num_layers=1
        )
        
        history = predictor.train(
            data=self.test_data,
            target_column='hvac_power',
            epochs=3,
            batch_size=16,
            verbose=False
        )
        
        self.assertGreater(len(history['train_loss']), 0)
    
    def test_gru_model(self):
        """测试GRU模型"""
        predictor = EnergyPredictor(
            model_type="gru",
            seq_length=12,
            hidden_size=32,
            num_layers=1
        )
        
        history = predictor.train(
            data=self.test_data,
            target_column='hvac_power',
            epochs=3,
            batch_size=16,
            verbose=False
        )
        
        self.assertGreater(len(history['train_loss']), 0)
    
    def test_mlp_model(self):
        """测试MLP模型"""
        predictor = EnergyPredictor(
            model_type="mlp",
            seq_length=12,
            hidden_size=32
        )
        
        history = predictor.train(
            data=self.test_data,
            target_column='hvac_power',
            epochs=3,
            batch_size=16,
            verbose=False
        )
        
        self.assertGreater(len(history['train_loss']), 0)


def run_tests():
    """运行所有测试"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestLSTMModel))
    suite.addTests(loader.loadTestsFromTestCase(TestGRUModel))
    suite.addTests(loader.loadTestsFromTestCase(TestMLPModel))
    suite.addTests(loader.loadTestsFromTestCase(TestTimeSeriesDataset))
    suite.addTests(loader.loadTestsFromTestCase(TestEnergyPredictor))
    suite.addTests(loader.loadTestsFromTestCase(TestHistoricalAverageBaseline))
    suite.addTests(loader.loadTestsFromTestCase(TestRegressionBaseline))
    suite.addTests(loader.loadTestsFromTestCase(TestClusteringBaseline))
    suite.addTests(loader.loadTestsFromTestCase(TestBaselineModel))
    suite.addTests(loader.loadTestsFromTestCase(TestModelTypes))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
