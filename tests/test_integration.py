"""
集成测试 - 验证所有模块可以正常导入和协作

任务ID: INTEG-D1-001
任务描述: 验证所有模块集成正常工作
优先级: High

测试内容:
1. 模块导入验证
2. 控制器初始化验证
3. 异常检测模块集成
4. 知识库模块集成
5. 预测模型集成
6. CLI命令验证
7. 配置文件验证
"""

import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestModuleImports(unittest.TestCase):
    """测试所有模块可以正常导入"""
    
    def test_import_building_simulator(self):
        """测试BuildingSimulator导入"""
        from building_energy.core.building_simulator import BuildingSimulator
        self.assertIsNotNone(BuildingSimulator)
    
    def test_import_hvac_env(self):
        """测试HVACEnv导入"""
        from building_energy.env.hvac_env import HVACEnv
        self.assertIsNotNone(HVACEnv)
    
    def test_import_weather_api(self):
        """测试WeatherAPI导入"""
        from building_energy.data.weather_api import WeatherAPI
        self.assertIsNotNone(WeatherAPI)
    
    def test_import_anomaly_detector(self):
        """测试AnomalyDetector导入"""
        from building_energy.models.anomaly_detector import AnomalyDetector
        self.assertIsNotNone(AnomalyDetector)
    
    def test_import_knowledge_base(self):
        """测试KnowledgeBase导入"""
        from building_energy.knowledge.graph_rag import KnowledgeBase
        self.assertIsNotNone(KnowledgeBase)
    
    def test_import_predictor(self):
        """测试EnergyPredictor导入"""
        from building_energy.models.predictor import EnergyPredictor
        self.assertIsNotNone(EnergyPredictor)
    
    def test_import_main_controller(self):
        """测试BuildingController导入"""
        from building_energy.main import BuildingController
        self.assertIsNotNone(BuildingController)
    
    def test_import_cli(self):
        """测试CLI导入"""
        from building_energy.cli import main, CLI
        self.assertIsNotNone(main)
        self.assertIsNotNone(CLI)


class TestControllerInitialization(unittest.TestCase):
    """测试主控制器可以初始化"""
    
    def test_controller_init_with_config_manager(self):
        """测试使用ConfigManager初始化控制器"""
        from building_energy.main import BuildingController
        from building_energy.config.manager import ConfigManager
        
        config = ConfigManager()
        controller = BuildingController(config)
        self.assertIsNotNone(controller)
        self.assertIsInstance(controller.config, ConfigManager)
    
    def test_controller_init_with_none(self):
        """测试使用None初始化控制器"""
        from building_energy.main import BuildingController
        
        controller = BuildingController(None)
        self.assertIsNotNone(controller)


class TestAnomalyDetectorIntegration(unittest.TestCase):
    """测试异常检测模块集成"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_detector_creation(self):
        """测试检测器创建"""
        from building_energy.models.anomaly_detector import AnomalyDetector
        
        detector = AnomalyDetector(algorithm="iforest")
        self.assertIsNotNone(detector)
        self.assertEqual(detector.algorithm, "iforest")
    
    def test_detector_fit_and_predict(self):
        """测试检测器训练和预测"""
        from building_energy.models.anomaly_detector import AnomalyDetector
        import numpy as np
        
        detector = AnomalyDetector(algorithm="iforest")
        
        # 模拟数据
        data = np.random.randn(100, 5)
        detector.fit(data)
        
        # 检测
        result = detector.predict(data[:10])
        self.assertEqual(len(result), 10)


class TestKnowledgeBaseIntegration(unittest.TestCase):
    """测试知识库模块集成"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_knowledge_base_creation(self):
        """测试知识库创建"""
        from building_energy.knowledge.graph_rag import KnowledgeBase
        
        kb = KnowledgeBase(root_dir=self.temp_dir)
        self.assertIsNotNone(kb)
    
    def test_knowledge_base_query(self):
        """测试知识库查询"""
        from building_energy.knowledge.graph_rag import KnowledgeBase, QueryResult
        
        kb = KnowledgeBase(root_dir=self.temp_dir)
        
        # 测试查询
        result = kb.query("如何节能？")
        self.assertIsInstance(result, QueryResult)
        self.assertIsInstance(result.answer, str)
        self.assertGreater(len(result.answer), 0)


class TestPredictorIntegration(unittest.TestCase):
    """测试预测模型集成"""
    
    def setUp(self):
        """测试前准备"""
        import pandas as pd
        import numpy as np
        
        # 创建模拟数据
        dates = pd.date_range('2024-01-01', periods=1000, freq='H')
        self.data = pd.DataFrame({
            'temperature': np.random.randn(1000) + 22,
            'humidity': np.random.randn(1000) + 50,
            'occupancy': np.random.randint(0, 50, 1000),
            'energy': np.random.randn(1000) + 100
        }, index=dates)
    
    def test_predictor_creation(self):
        """测试预测器创建"""
        from building_energy.models.predictor import EnergyPredictor
        
        predictor = EnergyPredictor(model_type="mlp", horizon=24)
        self.assertIsNotNone(predictor)
        self.assertEqual(predictor.model_type, "mlp")
    
    def test_predictor_train_and_predict(self):
        """测试预测器训练和预测"""
        from building_energy.models.predictor import EnergyPredictor, PredictionResult
        
        predictor = EnergyPredictor(model_type="mlp", horizon=24)
        predictor.train(self.data, epochs=5)
        
        # 预测 - predict方法需要传入data参数
        result = predictor.predict(data=self.data, horizon=24)
        self.assertIsInstance(result, PredictionResult)
        self.assertEqual(len(result.predictions), 24)


class TestCLIIntegration(unittest.TestCase):
    """测试CLI命令集成"""
    
    def test_cli_creation(self):
        """测试CLI创建"""
        from building_energy.cli import CLI
        
        cli = CLI()
        self.assertIsNotNone(cli)
    
    def test_cli_parser(self):
        """测试CLI参数解析器"""
        from building_energy.cli import CLI
        
        cli = CLI()
        self.assertIsNotNone(cli.parser)


class TestConfigIntegration(unittest.TestCase):
    """测试配置文件集成"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_config_manager_creation(self):
        """测试配置管理器创建"""
        from building_energy.config.manager import ConfigManager
        
        config = ConfigManager()
        self.assertIsNotNone(config)
    
    def test_default_config_exists(self):
        """测试默认配置文件存在"""
        config_path = Path(__file__).parent.parent / "building_energy" / "config" / "default_config.yaml"
        self.assertTrue(config_path.exists(), f"默认配置文件不存在: {config_path}")
    
    def test_default_config_yaml_valid(self):
        """测试默认配置文件YAML格式正确"""
        import yaml
        
        config_path = Path(__file__).parent.parent / "building_energy" / "config" / "default_config.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        self.assertIsNotNone(config)
        self.assertIn('system', config)
        self.assertIn('hvac', config)


class TestBuildingSimulatorIntegration(unittest.TestCase):
    """测试建筑模拟器集成"""
    
    def test_simulator_import(self):
        """测试模拟器导入"""
        from building_energy.core.building_simulator import BuildingSimulator
        self.assertIsNotNone(BuildingSimulator)


class TestHVACEnvIntegration(unittest.TestCase):
    """测试HVAC环境集成"""
    
    def test_hvac_env_import(self):
        """测试HVAC环境导入"""
        from building_energy.env.hvac_env import HVACEnv
        self.assertIsNotNone(HVACEnv)


class TestWeatherAPIIntegration(unittest.TestCase):
    """测试天气API集成"""
    
    def test_weather_api_import(self):
        """测试天气API导入"""
        from building_energy.data.weather_api import WeatherAPI
        self.assertIsNotNone(WeatherAPI)


def run_integration_tests():
    """运行所有集成测试"""
    print("=" * 60)
    print("集成测试 - 验证所有模块集成正常工作")
    print("任务ID: INTEG-D1-001")
    print("=" * 60)
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加所有测试类
    suite.addTests(loader.loadTestsFromTestCase(TestModuleImports))
    suite.addTests(loader.loadTestsFromTestCase(TestControllerInitialization))
    suite.addTests(loader.loadTestsFromTestCase(TestAnomalyDetectorIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestKnowledgeBaseIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestPredictorIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestCLIIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestBuildingSimulatorIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestHVACEnvIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestWeatherAPIIntegration))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 打印结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    print(f"总测试数: {result.testsRun}")
    print(f"通过: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ 所有集成测试通过！")
        return 0
    else:
        print("\n❌ 部分集成测试失败")
        return 1


if __name__ == "__main__":
    exit_code = run_integration_tests()
    sys.exit(exit_code)
