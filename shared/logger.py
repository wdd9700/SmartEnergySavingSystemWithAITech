#!/usr/bin/env python3
"""
共享日志模块
"""
import logging
import sys
from pathlib import Path
from datetime import datetime


LOG_FORMAT = '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


def setup_logger(name: str, log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """配置日志记录器"""
    
    # 创建日志目录
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 格式化
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    
    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件输出
    today = datetime.now().strftime('%Y%m%d')
    file_handler = logging.FileHandler(
        f"{log_dir}/{name}_{today}.log", encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger
