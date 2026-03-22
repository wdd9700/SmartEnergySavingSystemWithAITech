"""
进程监控与定期扫描系统

Module-3E: Process Scanner and Monitoring System

提供每10分钟一次的进程扫描功能，用于识别长时间运行的任务，防止误关机。

主要功能:
    - 进程列表扫描
    - 长时间运行任务识别
    - 进程白名单管理
    - 关机保护判断

使用示例:
    >>> from building_energy.scanner import ProcessScanner, get_scanner
    >>> 
    >>> # 创建扫描器实例
    >>> scanner = ProcessScanner()
    >>> 
    >>> # 执行扫描
    >>> long_tasks = scanner.scan()
    >>> for task in long_tasks:
    ...     print(f"长时间任务: {task.name}, 运行时间: {task.runtime_minutes:.1f}分钟")
    >>> 
    >>> # 检查是否应该阻止关机
    >>> should_block, reasons = scanner.should_prevent_shutdown()
    >>> if should_block:
    ...     print(f"检测到长时间任务，阻止关机: {reasons}")

作者: Smart Energy Saving System
版本: 1.0.0
"""

from .process_scanner import (
    ProcessScanner,
    ProcessInfo,
    ProcessType,
    get_scanner,
    reset_scanner,
)

from .whitelist import (
    WhitelistManager,
    WhitelistConfig,
    get_whitelist_manager,
    reset_whitelist_manager,
)

__version__ = "1.0.0"
__author__ = "Smart Energy Saving System"

__all__ = [
    # 进程扫描器
    "ProcessScanner",
    "ProcessInfo", 
    "ProcessType",
    "get_scanner",
    "reset_scanner",
    
    # 白名单管理
    "WhitelistManager",
    "WhitelistConfig",
    "get_whitelist_manager",
    "reset_whitelist_manager",
]


def create_scanner_from_config(config_path: str = None, **kwargs) -> ProcessScanner:
    """
    从配置文件创建扫描器实例
    
    Args:
        config_path: 配置文件路径，默认使用模块目录下的config.yaml
        **kwargs: 额外的配置参数，会覆盖配置文件中的值
        
    Returns:
        配置好的ProcessScanner实例
        
    Example:
        >>> scanner = create_scanner_from_config()
        >>> tasks = scanner.scan()
    """
    import os
    import yaml
    from pathlib import Path
    
    # 确定配置文件路径
    if config_path is None:
        module_dir = Path(__file__).parent
        config_path = module_dir / "config.yaml"
    
    # 默认配置
    scanner_config = {
        'long_running_threshold': 30,
        'scan_interval_minutes': 10,
    }
    
    # 加载配置文件
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                if data and 'scanner' in data:
                    scanner_config.update(data['scanner'])
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to load config: {e}")
    
    # 使用kwargs覆盖配置
    scanner_config.update(kwargs)
    
    return ProcessScanner(
        long_running_threshold=scanner_config.get('long_running_threshold', 30),
        scan_interval_minutes=scanner_config.get('scan_interval_minutes', 10),
    )


def quick_check() -> tuple[bool, list]:
    """
    快速检查是否应该阻止关机
    
    这是一个便捷函数，用于快速检查当前系统状态。
    它会自动创建扫描器并执行扫描。
    
    Returns:
        (是否阻止关机, 阻止原因列表)
        
    Example:
        >>> should_block, reasons = quick_check()
        >>> if should_block:
        ...     print("不应关机，原因:", reasons)
    """
    scanner = get_scanner()
    return scanner.should_prevent_shutdown()
