"""
HVAC强化学习训练脚本

使用Stable-Baselines3训练HVAC控制策略。

用法:
    # 训练SAC算法
    python train_hvac_rl.py --config config/default_config.yaml --algorithm SAC
    
    # 训练PPO算法
    python train_hvac_rl.py --config config/default_config.yaml --algorithm PPO
    
    # 训练TD3算法
    python train_hvac_rl.py --config config/default_config.yaml --algorithm TD3
    
    # 指定训练步数
    python train_hvac_rl.py --config config/default_config.yaml --algorithm SAC --timesteps 200000
    
    # 评估模型
    python train_hvac_rl.py --config config/default_config.yaml --mode evaluate --model output/models/hvac_sac_best.zip
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import yaml
import numpy as np
import gymnasium as gym
import torch
from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    ProgressBarCallback,
    CallbackList,
    BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.logger import configure

from core.building_simulator import BuildingSimulator
from env.hvac_env import HVACEnv

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 支持的算法映射
ALGORITHMS = {
    'SAC': SAC,
    'PPO': PPO,
    'TD3': TD3
}


class TensorBoardLoggingCallback(BaseCallback):
    """
    自定义TensorBoard日志回调
    记录额外的训练指标
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # 记录每步信息
        if self.locals.get('infos') and len(self.locals['infos']) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])
                    
                    # 记录到TensorBoard
                    self.logger.record("rollout/episode_reward", info['episode']['r'])
                    self.logger.record("rollout/episode_length", info['episode']['l'])
                    
                    if len(self.episode_rewards) >= 100:
                        avg_reward = np.mean(self.episode_rewards[-100:])
                        self.logger.record("rollout/avg_reward_100", avg_reward)
        
        return True


class BestModelSaveCallback(BaseCallback):
    """
    自动保存最佳模型回调
    基于评估奖励自动保存表现最好的模型
    """
    
    def __init__(self, eval_freq: int, save_path: str, verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.save_count = 0
        
    def _init_callback(self) -> None:
        # 创建保存目录
        if self.save_path is not None:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
    
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # 获取评估结果
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                # 尝试从日志获取评估奖励
                try:
                    mean_reward = self.model.logger.name_to_value.get('eval/mean_reward', None)
                    if mean_reward is not None:
                        self.last_mean_reward = mean_reward
                        
                        # 如果当前奖励更好，保存模型
                        if mean_reward > self.best_mean_reward:
                            self.best_mean_reward = mean_reward
                            self.save_count += 1
                            self.model.save(self.save_path)
                            if self.verbose > 0:
                                logger.info(
                                    f"New best model! Mean reward: {mean_reward:.2f} "
                                    f"(previous: {self.best_mean_reward:.2f})"
                                )
                                logger.info(f"Model saved to {self.save_path}")
                except Exception as e:
                    logger.warning(f"Could not get eval reward: {e}")
        
        return True


def make_env(config: dict, rank: int = 0, seed: int = 0):
    """
    创建环境工厂函数（支持向量化）
    
    Args:
        config: 配置字典
        rank: 环境编号（用于多进程）
        seed: 随机种子
    
    Returns:
        环境创建函数
    """
    def _init():
        # 创建建筑模拟器
        building_sim = BuildingSimulator(
            idf_path=config['simulation']['idf_path'],
            weather_path=config['simulation']['weather_path'],
            timestep=config['simulation']['timestep'],
            config=config
        )
        
        # 创建环境
        env = HVACEnv(building_sim=building_sim, config=config)
        
        # 设置种子
        env.reset(seed=seed + rank)
        
        # 包装Monitor用于记录统计
        log_dir = config['rl'].get('log_dir', 'output/rl_logs')
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, filename=f"{log_dir}/env_{rank}")
        
        return env
    
    return _init


def create_vec_env(
    config: dict, 
    n_envs: int = 1, 
    seed: int = 0,
    use_subprocess: bool = True
):
    """
    创建向量化环境
    
    支持DummyVecEnv（单进程）和SubprocVecEnv（多进程）
    
    Args:
        config: 配置字典
        n_envs: 并行环境数
        seed: 随机种子
        use_subprocess: 是否使用多进程（SubprocVecEnv）
    
    Returns:
        向量化环境
    """
    if n_envs == 1:
        # 单进程环境 - 使用DummyVecEnv
        logger.info("Creating single DummyVecEnv environment")
        env = DummyVecEnv([make_env(config, 0, seed)])
    else:
        # 多进程环境
        if use_subprocess:
            logger.info(f"Creating {n_envs} SubprocVecEnv environments")
            env = SubprocVecEnv(
                [make_env(config, i, seed) for i in range(n_envs)],
                start_method='spawn' if sys.platform == 'win32' else 'fork'
            )
        else:
            logger.info(f"Creating {n_envs} DummyVecEnv environments (sequential)")
            env = DummyVecEnv([make_env(config, i, seed) for i in range(n_envs)])
    
    return env


def get_algorithm_config(algorithm_name: str, config: dict) -> Dict[str, Any]:
    """
    获取算法特定配置
    
    Args:
        algorithm_name: 算法名称 (SAC, PPO, TD3)
        config: 配置字典
    
    Returns:
        算法参数字典
    """
    algorithm_name = algorithm_name.upper()
    rl_config = config.get('rl', {})
    
    # 基础配置
    base_config = {
        'policy': 'MlpPolicy',
        'verbose': 1,
        'learning_rate': rl_config.get('learning_rate', 3e-4),
        'tensorboard_log': rl_config.get('log_dir', 'output/rl_logs'),
        'device': rl_config.get('device', 'auto'),
    }
    
    # SAC算法配置
    if algorithm_name == 'SAC':
        base_config.update({
            'buffer_size': rl_config.get('buffer_size', 100000),
            'learning_starts': rl_config.get('learning_starts', 1000),
            'batch_size': rl_config.get('batch_size', 256),
            'tau': rl_config.get('tau', 0.005),
            'gamma': rl_config.get('gamma', 0.99),
            'train_freq': rl_config.get('train_freq', 1),
            'gradient_steps': rl_config.get('gradient_steps', 1),
            'ent_coef': rl_config.get('ent_coef', 'auto'),
        })
    
    # PPO算法配置
    elif algorithm_name == 'PPO':
        base_config.update({
            'n_steps': rl_config.get('n_steps', 2048),
            'batch_size': rl_config.get('batch_size', 64),
            'n_epochs': rl_config.get('n_epochs', 10),
            'gamma': rl_config.get('gamma', 0.99),
            'gae_lambda': rl_config.get('gae_lambda', 0.95),
            'clip_range': rl_config.get('clip_range', 0.2),
            'clip_range_vf': rl_config.get('clip_range_vf', None),
            'ent_coef': rl_config.get('ent_coef', 0.0),
            'vf_coef': rl_config.get('vf_coef', 0.5),
            'max_grad_norm': rl_config.get('max_grad_norm', 0.5),
            'use_sde': rl_config.get('use_sde', False),
            'sde_sample_freq': rl_config.get('sde_sample_freq', -1),
        })
    
    # TD3算法配置
    elif algorithm_name == 'TD3':
        base_config.update({
            'buffer_size': rl_config.get('buffer_size', 100000),
            'learning_starts': rl_config.get('learning_starts', 1000),
            'batch_size': rl_config.get('batch_size', 256),
            'tau': rl_config.get('tau', 0.005),
            'gamma': rl_config.get('gamma', 0.99),
            'train_freq': rl_config.get('train_freq', 1),
            'gradient_steps': rl_config.get('gradient_steps', 1),
            'action_noise': None,  # 可以添加动作噪声
            'policy_delay': rl_config.get('policy_delay', 2),
            'target_policy_noise': rl_config.get('target_policy_noise', 0.2),
            'target_noise_clip': rl_config.get('target_noise_clip', 0.5),
        })
    
    else:
        raise ValueError(f"不支持的算法: {algorithm_name}")
    
    return base_config


def setup_tensorboard_logger(log_dir: str, algorithm_name: str):
    """
    配置TensorBoard日志记录器
    
    Args:
        log_dir: 日志目录
        algorithm_name: 算法名称
    
    Returns:
        配置好的logger
    """
    tb_log_dir = os.path.join(log_dir, algorithm_name.lower())
    os.makedirs(tb_log_dir, exist_ok=True)
    
    # 配置TensorBoard输出格式
    logger_config = configure(
        tb_log_dir,
        ["stdout", "tensorboard"]
    )
    
    logger.info(f"TensorBoard logs will be saved to: {tb_log_dir}")
    logger.info(f"To view TensorBoard, run: tensorboard --logdir={tb_log_dir}")
    
    return logger_config


def train(args):
    """
    训练RL模型
    
    支持SAC、PPO、TD3三种算法，包含向量化环境、TensorBoard可视化和自动保存最佳模型
    
    Args:
        args: 命令行参数
    """
    start_time = time.time()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 更新配置
    if args.algorithm:
        config['rl']['algorithm'] = args.algorithm
    if args.timesteps:
        config['rl']['total_timesteps'] = args.timesteps
    if args.seed is not None:
        config['rl']['seed'] = args.seed
    
    # 获取配置
    rl_config = config['rl']
    algorithm_name = rl_config['algorithm'].upper()
    total_timesteps = rl_config['total_timesteps']
    n_envs = rl_config.get('n_envs', 1)
    seed = rl_config.get('seed', 0)
    log_dir = rl_config.get('log_dir', 'output/rl_logs')
    save_dir = rl_config.get('save_dir', 'output/models')
    eval_freq = rl_config.get('eval_freq', 5000)
    
    # 创建输出目录
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    logger.info("=" * 60)
    logger.info("HVAC强化学习训练开始")
    logger.info("=" * 60)
    logger.info(f"算法: {algorithm_name}")
    logger.info(f"并行环境数: {n_envs}")
    logger.info(f"训练步数: {total_timesteps}")
    logger.info(f"随机种子: {seed}")
    logger.info(f"日志目录: {log_dir}")
    logger.info(f"保存目录: {save_dir}")
    logger.info("=" * 60)
    
    # 创建向量化环境
    logger.info(f"创建 {n_envs} 个并行训练环境...")
    env = create_vec_env(config, n_envs=n_envs, seed=seed)
    
    # 创建评估环境
    logger.info("创建评估环境...")
    eval_env = create_vec_env(config, n_envs=1, seed=seed + 1000)
    
    # 获取算法配置
    algorithm_kwargs = get_algorithm_config(algorithm_name, config)
    algorithm_kwargs['env'] = env
    
    # 配置TensorBoard
    tb_logger = setup_tensorboard_logger(log_dir, algorithm_name)
    
    # 创建模型
    logger.info(f"初始化 {algorithm_name} 模型...")
    if algorithm_name not in ALGORITHMS:
        raise ValueError(f"不支持的算法: {algorithm_name}。支持的算法: {list(ALGORITHMS.keys())}")
    
    model_class = ALGORITHMS[algorithm_name]
    model = model_class(**algorithm_kwargs)
    model.set_logger(tb_logger)
    
    # 记录模型架构信息
    logger.info(f"模型策略网络: {algorithm_kwargs['policy']}")
    logger.info(f"学习率: {algorithm_kwargs['learning_rate']}")
    logger.info(f"折扣因子: {algorithm_kwargs['gamma']}")
    
    # 构建回调函数列表
    callbacks = []
    
    # 1. 评估回调 - 定期评估模型性能
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
        verbose=1
    )
    callbacks.append(eval_callback)
    
    # 2. 检查点回调 - 定期保存模型检查点
    checkpoint_callback = CheckpointCallback(
        save_freq=max(eval_freq, 10000),  # 至少每10000步保存一次
        save_path=os.path.join(save_dir, "checkpoints"),
        name_prefix=f"hvac_{algorithm_name.lower()}_checkpoint"
    )
    callbacks.append(checkpoint_callback)
    
    # 3. 自定义最佳模型保存回调
    best_model_path = os.path.join(save_dir, f"hvac_{algorithm_name.lower()}_best.zip")
    best_model_callback = BestModelSaveCallback(
        eval_freq=eval_freq,
        save_path=best_model_path,
        verbose=1
    )
    callbacks.append(best_model_callback)
    
    # 4. TensorBoard详细日志回调
    tb_callback = TensorBoardLoggingCallback(verbose=0)
    callbacks.append(tb_callback)
    
    # 5. 进度条回调
    callbacks.append(ProgressBarCallback())
    
    # 组合所有回调
    callback_list = CallbackList(callbacks)
    
    # 开始训练
    logger.info("=" * 60)
    logger.info(f"开始训练 {total_timesteps} 步...")
    logger.info("=" * 60)
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            progress_bar=True,
            reset_num_timesteps=True
        )
        
        # 保存最终模型
        final_model_path = os.path.join(save_dir, f"hvac_{algorithm_name.lower()}_final.zip")
        model.save(final_model_path)
        logger.info(f"最终模型已保存到: {final_model_path}")
        
        # 保存最佳模型（如果没有保存过）
        if best_model_callback.save_count == 0:
            model.save(best_model_path)
            logger.info(f"最佳模型已保存到: {best_model_path}")
        
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
        # 保存中断时的模型
        interrupt_path = os.path.join(save_dir, f"hvac_{algorithm_name.lower()}_interrupted.zip")
        model.save(interrupt_path)
        logger.info(f"中断时的模型已保存到: {interrupt_path}")
    
    finally:
        # 关闭环境
        env.close()
        eval_env.close()
    
    # 训练完成统计
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    
    logger.info("=" * 60)
    logger.info("训练完成!")
    logger.info(f"总训练时间: {hours}小时 {minutes}分钟 {seconds}秒")
    logger.info(f"模型保存位置: {save_dir}")
    logger.info(f"  - 最终模型: hvac_{algorithm_name.lower()}_final.zip")
    logger.info(f"  - 最佳模型: hvac_{algorithm_name.lower()}_best.zip")
    logger.info(f"TensorBoard日志: {log_dir}")
    logger.info(f"查看命令: tensorboard --logdir={log_dir}")
    logger.info("=" * 60)


def evaluate(args):
    """
    评估训练好的模型
    
    支持评估单个模型或比较多个模型
    
    Args:
        args: 命令行参数
    """
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 确定算法
    algorithm_name = args.algorithm or config['rl']['algorithm'].upper()
    
    # 确定模型路径
    save_dir = config['rl'].get('save_dir', 'output/models')
    if args.model:
        model_paths = [args.model]
    else:
        # 默认评估最佳模型
        model_paths = [
            os.path.join(save_dir, f"hvac_{algorithm_name.lower()}_best.zip"),
            os.path.join(save_dir, f"hvac_{algorithm_name.lower()}_final.zip")
        ]
    
    # 创建环境
    logger.info("创建评估环境...")
    env = create_vec_env(config, n_envs=1, seed=args.seed or 42)
    
    results = []
    
    for model_path in model_paths:
        if not os.path.exists(model_path):
            logger.warning(f"模型文件不存在: {model_path}")
            continue
        
        logger.info("=" * 60)
        logger.info(f"评估模型: {model_path}")
        logger.info("=" * 60)
        
        # 加载模型
        if algorithm_name not in ALGORITHMS:
            raise ValueError(f"不支持的算法: {algorithm_name}")
        
        model_class = ALGORITHMS[algorithm_name]
        try:
            model = model_class.load(model_path, env=env)
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            continue
        
        # 评估
        episode_rewards = []
        episode_lengths = []
        episode_comfort_violations = []
        episode_energy_consumptions = []
        
        for episode in range(args.episodes):
            obs, info = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            comfort_violations = 0
            energy_consumption = 0
            
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                
                # 从info获取额外统计
                if isinstance(info, dict):
                    if 'comfort_violation' in info:
                        comfort_violations += info['comfort_violation']
                    if 'hvac_power' in info:
                        energy_consumption += info['hvac_power']
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_comfort_violations.append(comfort_violations)
            episode_energy_consumptions.append(energy_consumption)
            
            logger.info(
                f"回合 {episode + 1}/{args.episodes}: "
                f"奖励={episode_reward:.2f}, "
                f"长度={episode_length}"
            )
        
        # 统计结果
        result = {
            'model_path': model_path,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'mean_comfort_violations': np.mean(episode_comfort_violations) if episode_comfort_violations else 0,
            'mean_energy_consumption': np.mean(episode_energy_consumptions) if episode_energy_consumptions else 0,
        }
        results.append(result)
        
        logger.info("-" * 60)
        logger.info(f"评估结果 ({os.path.basename(model_path)}):")
        logger.info(f"  平均奖励: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
        logger.info(f"  平均回合长度: {result['mean_length']:.2f}")
        if result['mean_comfort_violations'] > 0:
            logger.info(f"  平均舒适度违规: {result['mean_comfort_violations']:.2f}")
        if result['mean_energy_consumption'] > 0:
            logger.info(f"  平均能耗: {result['mean_energy_consumption']:.2f} kWh")
        logger.info("-" * 60)
    
    # 关闭环境
    env.close()
    
    # 输出比较结果
    if len(results) > 1:
        logger.info("=" * 60)
        logger.info("模型比较:")
        logger.info("=" * 60)
        best_model = max(results, key=lambda x: x['mean_reward'])
        for result in results:
            marker = " ★最佳" if result == best_model else ""
            logger.info(
                f"{os.path.basename(result['model_path'])}: "
                f"{result['mean_reward']:.2f} ± {result['std_reward']:.2f}{marker}"
            )
    
    return results


def main():
    """
    主函数 - 命令行入口
    
    支持命令:
        train: 训练模型
        evaluate: 评估模型
    
    示例:
        python train_hvac_rl.py train --algorithm SAC --timesteps 100000
        python train_hvac_rl.py evaluate --model output/models/hvac_sac_best.zip --episodes 20
    """
    parser = argparse.ArgumentParser(
        description='HVAC强化学习训练与评估工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 训练SAC算法
  python train_hvac_rl.py train --algorithm SAC --timesteps 100000
  
  # 训练PPO算法
  python train_hvac_rl.py train --algorithm PPO --timesteps 100000
  
  # 训练TD3算法
  python train_hvac_rl.py train --algorithm TD3 --timesteps 100000
  
  # 评估最佳模型
  python train_hvac_rl.py evaluate --model output/models/hvac_sac_best.zip
  
  # 评估并指定回合数
  python train_hvac_rl.py evaluate --episodes 20
  
  # 查看TensorBoard
  tensorboard --logdir=output/rl_logs
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        default='building_energy/config/default_config.yaml',
        help='配置文件路径 (默认: building_energy/config/default_config.yaml)'
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        choices=['SAC', 'PPO', 'TD3'],
        help='RL算法 (覆盖配置文件设置)'
    )
    parser.add_argument(
        '--timesteps',
        type=int,
        help='训练总步数 (覆盖配置文件设置)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='随机种子 (覆盖配置文件设置)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令', title='命令')
    
    # 训练命令
    train_parser = subparsers.add_parser(
        'train',
        help='训练RL模型',
        description='使用指定算法训练HVAC控制策略'
    )
    train_parser.set_defaults(func=train)
    
    # 评估命令
    eval_parser = subparsers.add_parser(
        'evaluate',
        help='评估训练好的模型',
        description='评估已训练模型的性能'
    )
    eval_parser.add_argument(
        '--model',
        type=str,
        help='模型文件路径 (默认: 使用配置中的最佳模型)'
    )
    eval_parser.add_argument(
        '--episodes',
        type=int,
        default=10,
        help='评估回合数 (默认: 10)'
    )
    eval_parser.set_defaults(func=evaluate)
    
    args = parser.parse_args()
    
    if args.command is None:
        # 如果没有指定命令，默认执行训练
        args.command = 'train'
        args.func = train
    
    try:
        args.func(args)
    except Exception as e:
        logger.error(f"执行失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
