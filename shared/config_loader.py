#!/usr/bin/env python3
"""
配置加载工具
支持YAML和命令行参数覆盖
"""
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str, defaults: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
        defaults: 默认配置值
    
    Returns:
        合并后的配置字典
    """
    config = defaults or {}
    
    if Path(config_path).exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            file_config = yaml.safe_load(f)
            if file_config:
                config.update(file_config)
    
    return config


def merge_config(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    用命令行参数覆盖配置
    
    Args:
        config: 配置文件加载的配置
        args: 命令行参数
    
    Returns:
        合并后的配置
    """
    args_dict = vars(args)
    
    # 只覆盖非None的值
    for key, value in args_dict.items():
        if value is not None:
            config[key] = value
    
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """保存配置到文件"""
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
