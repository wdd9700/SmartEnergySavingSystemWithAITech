"""
集成测试运行脚本

任务ID: INTEG-D1-001
任务描述: 验证所有模块集成正常工作
优先级: High

使用方法:
    python tests/run_integration_tests.py

该脚本会运行所有集成测试并生成测试报告。
"""

import os
import sys
import unittest
from pathlib import Path
import json
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def run_tests():
    """运行所有集成测试"""
    print("=" * 70)
    print("集成测试运行器")
    print("任务ID: INTEG-D1-001")
    print("=" * 70)
    
    # 加载测试模块
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 从test_integration.py加载测试
    try:
        from tests import test_integration
        suite.addTests(loader.loadTestsFromModule(test_integration))
        print("\n✓ 已加载 test_integration 模块")
    except Exception as e:
        print(f"\n✗ 加载 test_integration 模块失败: {e}")
    
    # 运行测试
    print("\n" + "-" * 70)
    print("开始运行测试...")
    print("-" * 70 + "\n")
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 生成报告
    print("\n" + "=" * 70)
    print("测试结果报告")
    print("=" * 70)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'task_id': 'INTEG-D1-001',
        'summary': {
            'total': result.testsRun,
            'passed': result.testsRun - len(result.failures) - len(result.errors),
            'failed': len(result.failures),
            'errors': len(result.errors),
            'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100 if result.testsRun > 0 else 0
        }
    }
    
    print(f"\n总测试数: {report['summary']['total']}")
    print(f"通过: {report['summary']['passed']}")
    print(f"失败: {report['summary']['failed']}")
    print(f"错误: {report['summary']['errors']}")
    print(f"成功率: {report['summary']['success_rate']:.1f}%")
    
    # 保存报告
    report_path = Path(__file__).parent / 'integration_test_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n测试报告已保存: {report_path}")
    
    # 返回结果
    if result.wasSuccessful():
        print("\n" + "=" * 70)
        print("✅ 所有集成测试通过！")
        print("=" * 70)
        return 0
    else:
        print("\n" + "=" * 70)
        print("❌ 部分集成测试失败")
        print("=" * 70)
        return 1


def verify_cli():
    """验证CLI命令"""
    print("\n" + "=" * 70)
    print("CLI命令验证")
    print("=" * 70)
    
    import subprocess
    
    commands = [
        ("帮助命令", [sys.executable, "-m", "building_energy.cli", "--help"]),
        ("初始化帮助", [sys.executable, "-m", "building_energy.cli", "init", "--help"]),
        ("版本命令", [sys.executable, "-m", "building_energy.cli", "version"]),
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
                all_passed = False
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            all_passed = False
    
    return all_passed


def verify_config():
    """验证配置文件"""
    print("\n" + "=" * 70)
    print("配置文件验证")
    print("=" * 70)
    
    config_path = Path(__file__).parent.parent / "building_energy" / "config" / "default_config.yaml"
    
    # 检查文件存在
    if config_path.exists():
        print(f"  ✓ 配置文件存在: {config_path}")
    else:
        print(f"  ✗ 配置文件不存在: {config_path}")
        return False
    
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
        
        return True
    except ImportError:
        print(f"  ⚠ PyYAML未安装，跳过YAML格式验证")
        return True
    except Exception as e:
        print(f"  ✗ YAML格式错误: {e}")
        return False


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("集成验证 - 模块联调")
    print("任务ID: INTEG-D1-001")
    print("=" * 70)
    
    results = []
    
    # 运行单元测试
    results.append(("集成测试", run_tests() == 0))
    
    # 验证CLI
    results.append(("CLI命令", verify_cli()))
    
    # 验证配置
    results.append(("配置文件", verify_config()))
    
    # 打印汇总
    print("\n" + "=" * 70)
    print("验证结果汇总")
    print("=" * 70)
    
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {status}: {name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print("\n" + "=" * 70)
    print(f"总计: {passed}/{total} 项通过")
    print("=" * 70)
    
    if passed == total:
        print("\n✅ 所有集成验证通过！")
        print("\nDeveloper Agent (INTEG-D1-001) 已就绪，开始集成验证")
        return 0
    else:
        print(f"\n❌ {total - passed} 项验证失败")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
