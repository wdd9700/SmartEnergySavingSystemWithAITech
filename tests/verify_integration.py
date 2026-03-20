"""
集成验证脚本

任务ID: INTEG-D1-001
任务描述: 验证所有模块集成正常工作
优先级: High

验证内容:
1. 模块导入验证
2. CLI命令验证
3. 配置文件验证
"""

import os
import sys
import subprocess
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def print_section(title):
    """打印章节标题"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def verify_module_imports():
    """验证所有模块可以正常导入"""
    print_section("1. 模块导入验证")
    
    modules = [
        ("building_energy.core.building_simulator", "BuildingSimulator"),
        ("building_energy.env.hvac_env", "HVACEnv"),
        ("building_energy.data.weather_api", "WeatherAPI"),
        ("building_energy.models.anomaly_detector", "AnomalyDetector"),
        ("building_energy.knowledge.graph_rag", "KnowledgeBase"),
        ("building_energy.models.predictor", "EnergyPredictor"),
        ("building_energy.main", "BuildingController"),
        ("building_energy.cli", "main"),
        ("building_energy.config.manager", "ConfigManager"),
    ]
    
    all_passed = True
    for module_name, class_name in modules:
        try:
            exec(f"from {module_name} import {class_name}")
            print(f"  ✓ {module_name}.{class_name}")
        except Exception as e:
            print(f"  ✗ {module_name}.{class_name}: {e}")
            all_passed = False
    
    return all_passed


def verify_cli_commands():
    """验证CLI命令可以正常执行"""
    print_section("2. CLI命令验证")
    
    commands = [
        ("帮助命令", ["python", "-m", "building_energy.cli", "--help"]),
        ("初始化帮助", ["python", "-m", "building_energy.cli", "init", "--help"]),
        ("版本命令", ["python", "-m", "building_energy.cli", "version"]),
    ]
    
    all_passed = True
    for name, cmd in commands:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
                cwd=os.path.dirname(os.path.dirname(__file__))
            )
            if result.returncode == 0:
                print(f"  ✓ {name}")
            else:
                print(f"  ✗ {name}: 返回码 {result.returncode}")
                if result.stderr:
                    print(f"    错误: {result.stderr[:200]}")
                all_passed = False
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            all_passed = False
    
    return all_passed


def verify_config_file():
    """验证配置文件完整"""
    print_section("3. 配置文件验证")
    
    all_passed = True
    
    # 检查配置文件存在
    config_path = Path(__file__).parent.parent / "building_energy" / "config" / "default_config.yaml"
    if config_path.exists():
        print(f"  ✓ 配置文件存在: {config_path}")
    else:
        print(f"  ✗ 配置文件不存在: {config_path}")
        all_passed = False
        return all_passed
    
    # 验证YAML格式
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"  ✓ YAML格式正确")
        
        # 检查必要配置项
        required_sections = ['system', 'hvac', 'simulation', 'rl']
        for section in required_sections:
            if section in config:
                print(f"  ✓ 配置段 '{section}' 存在")
            else:
                print(f"  ✗ 配置段 '{section}' 缺失")
                all_passed = False
    
    except ImportError:
        print(f"  ⚠ PyYAML未安装，跳过YAML格式验证")
    except Exception as e:
        print(f"  ✗ YAML格式错误: {e}")
        all_passed = False
    
    return all_passed


def verify_controller_initialization():
    """验证主控制器可以初始化"""
    print_section("4. 控制器初始化验证")
    
    try:
        from building_energy.main import BuildingController
        from building_energy.config.manager import ConfigManager
        
        config = ConfigManager()
        controller = BuildingController(config)
        
        print(f"  ✓ BuildingController可以初始化")
        return True
    except Exception as e:
        print(f"  ✗ BuildingController初始化失败: {e}")
        return False


def verify_anomaly_detector():
    """验证异常检测模块集成"""
    print_section("5. 异常检测模块集成验证")
    
    try:
        from building_energy.models.anomaly_detector import AnomalyDetector
        import numpy as np
        
        # 创建检测器
        detector = AnomalyDetector(algorithm="iforest")
        
        # 模拟数据
        data = np.random.randn(100, 5)
        detector.fit(data)
        
        # 检测
        result = detector.predict(data[:10])
        
        # 验证结果
        assert len(result) == 10, f"期望10个预测结果，实际得到 {len(result)}"
        
        print(f"  ✓ AnomalyDetector可以正常训练和预测")
        return True
    except Exception as e:
        print(f"  ✗ AnomalyDetector集成失败: {e}")
        return False


def verify_knowledge_base():
    """验证知识库模块集成"""
    print_section("6. 知识库模块集成验证")
    
    try:
        from building_energy.knowledge.graph_rag import KnowledgeBase, QueryResult
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            kb = KnowledgeBase(root_dir=temp_dir)
            result = kb.query("如何节能？")
            
            # 验证返回的是QueryResult对象
            assert isinstance(result, QueryResult), f"期望QueryResult，实际得到 {type(result)}"
            assert isinstance(result.answer, str), f"期望answer是字符串，实际得到 {type(result.answer)}"
            
            print(f"  ✓ KnowledgeBase可以正常查询")
            return True
    except Exception as e:
        print(f"  ✗ KnowledgeBase集成失败: {e}")
        return False


def verify_predictor():
    """验证预测模型集成"""
    print_section("7. 预测模型集成验证")
    
    try:
        from building_energy.models.predictor import EnergyPredictor, PredictionResult
        import pandas as pd
        import numpy as np
        
        # 创建模拟数据
        dates = pd.date_range('2024-01-01', periods=1000, freq='H')
        data = pd.DataFrame({
            'temperature': np.random.randn(1000) + 22,
            'humidity': np.random.randn(1000) + 50,
            'occupancy': np.random.randint(0, 50, 1000),
            'energy': np.random.randn(1000) + 100
        }, index=dates)
        
        # 创建预测器
        predictor = EnergyPredictor(model_type="mlp", horizon=24)
        predictor.train(data, epochs=5)
        
        # 预测 - predict方法需要传入data参数
        result = predictor.predict(data=data, horizon=24)
        
        # 验证返回的是PredictionResult对象
        assert isinstance(result, PredictionResult), f"期望PredictionResult，实际得到 {type(result)}"
        assert len(result.predictions) == 24, f"期望24个预测值，实际得到 {len(result.predictions)}"
        
        print(f"  ✓ EnergyPredictor可以正常训练和预测")
        return True
    except Exception as e:
        print(f"  ✗ EnergyPredictor集成失败: {e}")
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("集成验证 - 模块联调")
    print("任务ID: INTEG-D1-001")
    print("=" * 60)
    
    results = []
    
    # 运行所有验证
    results.append(("模块导入", verify_module_imports()))
    results.append(("CLI命令", verify_cli_commands()))
    results.append(("配置文件", verify_config_file()))
    results.append(("控制器初始化", verify_controller_initialization()))
    results.append(("异常检测模块", verify_anomaly_detector()))
    results.append(("知识库模块", verify_knowledge_base()))
    results.append(("预测模型", verify_predictor()))
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("验证结果汇总")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {status}: {name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print("\n" + "=" * 60)
    print(f"总计: {passed}/{total} 项通过")
    print("=" * 60)
    
    if passed == total:
        print("\n✅ 所有集成验证通过！")
        return 0
    else:
        print(f"\n❌ {total - passed} 项验证失败")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
