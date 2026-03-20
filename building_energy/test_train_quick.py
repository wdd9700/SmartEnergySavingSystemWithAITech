"""
快速测试RL训练流程

用于验证训练脚本的基本功能，使用简化配置快速运行。
"""

import os
import sys
import tempfile
import yaml

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_test_config():
    """创建测试配置"""
    config = {
        'simulation': {
            'idf_path': 'models/building.idf',
            'weather_path': 'weather/CHN_Beijing.Beijing.545110_CSWD.epw',
            'timestep': 15,
            'simulation_days': 1,
            'output_dir': 'output/simulation',
            'verbose': False
        },
        'hvac': {
            'control_mode': 'temperature',
            'min_setpoint': 18.0,
            'max_setpoint': 26.0,
            'target_temperature': 22.0,
            'comfort_tolerance': 1.0,
            'hvac_power': 5.0,
            'natural_ventilation': True
        },
        'rl': {
            'algorithm': 'SAC',
            'total_timesteps': 1000,  # 快速测试用少量步数
            'seed': 42,
            'learning_rate': 0.0003,
            'batch_size': 64,
            'buffer_size': 10000,
            'gamma': 0.99,
            'tau': 0.005,
            'train_freq': 1,
            'gradient_steps': 1,
            'learning_starts': 100,
            'ent_coef': 'auto',
            # PPO配置
            'n_steps': 128,
            'n_epochs': 4,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            # TD3配置
            'policy_delay': 2,
            'target_policy_noise': 0.2,
            'target_noise_clip': 0.5,
            # 训练配置
            'log_dir': 'output/test_rl_logs',
            'save_dir': 'output/test_models',
            'eval_freq': 500,
            'n_eval_episodes': 2,
            'n_envs': 1,
            'device': 'cpu'
        },
        'reward': {
            'comfort_weight': 1.0,
            'energy_weight': 0.1,
            'temp_penalty_coeff': 2.0,
            'max_reward': 0.0
        }
    }
    return config


def test_algorithm(algorithm_name: str, timesteps: int = 500):
    """测试指定算法"""
    print(f"\n{'='*60}")
    print(f"测试算法: {algorithm_name}")
    print(f"{'='*60}\n")
    
    # 创建临时配置文件
    config = create_test_config()
    config['rl']['algorithm'] = algorithm_name
    config['rl']['total_timesteps'] = timesteps
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name
    
    try:
        # 导入训练模块
        from train_hvac_rl import train, ALGORITHMS
        import argparse
        
        # 检查算法支持
        if algorithm_name not in ALGORITHMS:
            print(f"错误: 不支持的算法 {algorithm_name}")
            return False
        
        # 创建参数
        class Args:
            config = config_path
            algorithm = algorithm_name
            timesteps = timesteps
            seed = 42
        
        args = Args()
        
        # 运行训练
        train(args)
        
        print(f"\n✓ {algorithm_name} 算法测试通过!")
        return True
        
    except Exception as e:
        print(f"\n✗ {algorithm_name} 算法测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # 清理临时文件
        os.unlink(config_path)


def main():
    """主函数"""
    print("="*60)
    print("HVAC RL训练流程快速测试")
    print("="*60)
    
    results = {}
    
    # 测试SAC算法
    results['SAC'] = test_algorithm('SAC', timesteps=500)
    
    # 测试PPO算法
    results['PPO'] = test_algorithm('PPO', timesteps=500)
    
    # 测试TD3算法
    results['TD3'] = test_algorithm('TD3', timesteps=500)
    
    # 输出测试结果汇总
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    for algorithm, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {algorithm}: {status}")
    
    all_passed = all(results.values())
    print("="*60)
    
    if all_passed:
        print("✓ 所有算法测试通过!")
        return 0
    else:
        print("✗ 部分算法测试失败")
        return 1


if __name__ == '__main__':
    sys.exit(main())
