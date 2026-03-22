"""
白名单管理系统

提供系统进程白名单和用户自定义白名单的管理功能
"""

import os
import json
import yaml
import logging
from typing import Set, List, Dict, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class WhitelistConfig:
    """白名单配置数据类"""
    system_processes: Set[str]
    user_processes: Set[str]
    patterns: List[str]  # 正则表达式模式
    auto_update: bool = True
    last_updated: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "system_processes": sorted(list(self.system_processes)),
            "user_processes": sorted(list(self.user_processes)),
            "patterns": self.patterns,
            "auto_update": self.auto_update,
            "last_updated": self.last_updated,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WhitelistConfig":
        """从字典创建"""
        return cls(
            system_processes=set(data.get("system_processes", [])),
            user_processes=set(data.get("user_processes", [])),
            patterns=data.get("patterns", []),
            auto_update=data.get("auto_update", True),
            last_updated=data.get("last_updated"),
        )


class WhitelistManager:
    """
    白名单管理器
    
    管理系统进程白名单和用户自定义白名单，支持持久化存储
    """
    
    # Windows系统核心进程
    DEFAULT_SYSTEM_WHITELIST: Set[str] = {
        # Windows系统核心
        'svchost.exe', 'csrss.exe', 'smss.exe', 'services.exe',
        'lsass.exe', 'winlogon.exe', 'explorer.exe', 'dwm.exe',
        'system', 'registry', 'memory compression', 'system interrupts',
        'idle', 'crss.exe', 'wininit.exe', 'fontdrvhost.exe',
        'dllhost.exe', 'wmiprvse.exe', 'runtimebroker.exe',
        'sihost.exe', 'taskhostw.exe', 'shellhost.exe',
        'searchindexer.exe', 'searchfilterhost.exe', 'searchprotocolhost.exe',
        
        # Windows服务
        'spoolsv.exe', 'msmpeng.exe', 'nissrv.exe', 'securityhealthservice.exe',
        'wlanext.exe', 'conhost.exe', 'consent.exe',
        
        # 图形界面
        'dwm.exe', 'uxsms.exe', 'rdpclip.exe', 'csrss.exe',
        
        # 系统工具
        'taskmgr.exe', 'cmd.exe', 'powershell.exe', 'pwsh.exe',
    }
    
    # Linux系统核心进程
    LINUX_SYSTEM_WHITELIST: Set[str] = {
        'systemd', 'init', 'kthreadd', 'ksoftirqd', 'kworker',
        'migration', 'rcu_', 'watchdog', 'cpuhp', 'kdevtmpfs',
        'kauditd', 'khungtaskd', 'oom_reaper', 'kcompactd',
        'ksmd', 'khugepaged', 'kintegrityd', 'kblockd',
        'kswapd', 'nfsd', 'rpcbind', 'dbus-daemon',
        'systemd-journald', 'systemd-logind', 'systemd-networkd',
        'sshd', 'cron', 'rsyslogd', 'NetworkManager',
    }
    
    # Python相关（如果运行的是节能系统本身）
    PYTHON_WHITELIST: Set[str] = {
        'python.exe', 'python3.exe', 'python', 'python3',
        'pythonw.exe', 'py.exe',
    }
    
    # 默认保护模式
    DEFAULT_PATTERNS: List[str] = [
        r'.*system.*',  # 包含system的进程
        r'.*service.*',  # 包含service的进程
        r'.*daemon.*',  # 包含daemon的进程（Linux）
    ]
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化白名单管理器
        
        Args:
            config_path: 配置文件路径，默认使用模块目录下的config.yaml
        """
        self._config_path = config_path or self._get_default_config_path()
        self._config = self._load_or_create_config()
        
        logger.info(f"WhitelistManager initialized with {len(self._config.system_processes)} "
                   f"system and {len(self._config.user_processes)} user processes")
    
    def _get_default_config_path(self) -> str:
        """获取默认配置文件路径"""
        module_dir = Path(__file__).parent
        return str(module_dir / "config.yaml")
    
    def _load_or_create_config(self) -> WhitelistConfig:
        """加载或创建默认配置"""
        if os.path.exists(self._config_path):
            try:
                with open(self._config_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    if data and 'whitelist' in data:
                        logger.info(f"Loaded whitelist config from {self._config_path}")
                        return WhitelistConfig.from_dict(data['whitelist'])
            except Exception as e:
                logger.warning(f"Failed to load config from {self._config_path}: {e}")
        
        # 创建默认配置
        config = WhitelistConfig(
            system_processes=self._get_default_system_whitelist(),
            user_processes=set(),
            patterns=self.DEFAULT_PATTERNS.copy(),
        )
        self._save_config(config)
        return config
    
    def _get_default_system_whitelist(self) -> Set[str]:
        """获取默认系统白名单（根据操作系统）"""
        import platform
        
        system = platform.system().lower()
        base_whitelist = self.PYTHON_WHITELIST.copy()
        
        if system == 'windows':
            return base_whitelist | self.DEFAULT_SYSTEM_WHITELIST
        elif system == 'linux':
            return base_whitelist | self.LINUX_SYSTEM_WHITELIST
        else:
            return base_whitelist
    
    def _save_config(self, config: WhitelistConfig) -> None:
        """保存配置到文件"""
        try:
            from datetime import datetime
            config.last_updated = datetime.now().isoformat()
            
            data = {'whitelist': config.to_dict()}
            
            # 确保目录存在
            os.makedirs(os.path.dirname(self._config_path), exist_ok=True)
            
            with open(self._config_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            
            logger.debug(f"Saved whitelist config to {self._config_path}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def is_whitelisted(self, process_name: str) -> bool:
        """
        检查进程是否在白名单中
        
        Args:
            process_name: 进程名
            
        Returns:
            是否在白名单中
        """
        if not process_name:
            return False
        
        name_lower = process_name.lower()
        
        # 检查系统白名单
        if name_lower in {p.lower() for p in self._config.system_processes}:
            return True
        
        # 检查用户白名单
        if name_lower in {p.lower() for p in self._config.user_processes}:
            return True
        
        # 检查模式匹配
        import re
        for pattern in self._config.patterns:
            try:
                if re.match(pattern, process_name, re.IGNORECASE):
                    return True
            except re.error:
                continue
        
        return False
    
    def add_user_process(self, process_name: str, save: bool = True) -> bool:
        """
        添加用户自定义进程到白名单
        
        Args:
            process_name: 进程名
            save: 是否立即保存到文件
            
        Returns:
            是否成功添加
        """
        if not process_name:
            return False
        
        if process_name in self._config.user_processes:
            logger.debug(f"Process {process_name} already in whitelist")
            return True
        
        self._config.user_processes.add(process_name)
        logger.info(f"Added {process_name} to user whitelist")
        
        if save:
            self._save_config(self._config)
        
        return True
    
    def remove_user_process(self, process_name: str, save: bool = True) -> bool:
        """
        从用户白名单中移除进程
        
        Args:
            process_name: 进程名
            save: 是否立即保存到文件
            
        Returns:
            是否成功移除
        """
        if process_name in self._config.user_processes:
            self._config.user_processes.remove(process_name)
            logger.info(f"Removed {process_name} from user whitelist")
            
            if save:
                self._save_config(self._config)
            
            return True
        
        return False
    
    def add_pattern(self, pattern: str, save: bool = True) -> bool:
        """
        添加正则表达式模式
        
        Args:
            pattern: 正则表达式
            save: 是否立即保存到文件
            
        Returns:
            是否成功添加
        """
        import re
        
        # 验证正则表达式
        try:
            re.compile(pattern)
        except re.error as e:
            logger.error(f"Invalid regex pattern '{pattern}': {e}")
            return False
        
        if pattern in self._config.patterns:
            return True
        
        self._config.patterns.append(pattern)
        logger.info(f"Added pattern '{pattern}' to whitelist")
        
        if save:
            self._save_config(self._config)
        
        return True
    
    def remove_pattern(self, pattern: str, save: bool = True) -> bool:
        """
        移除正则表达式模式
        
        Args:
            pattern: 正则表达式
            save: 是否立即保存到文件
            
        Returns:
            是否成功移除
        """
        if pattern in self._config.patterns:
            self._config.patterns.remove(pattern)
            logger.info(f"Removed pattern '{pattern}' from whitelist")
            
            if save:
                self._save_config(self._config)
            
            return True
        
        return False
    
    def get_all_whitelist(self) -> Set[str]:
        """
        获取所有白名单进程（系统+用户）
        
        Returns:
            所有白名单进程集合
        """
        return self._config.system_processes | self._config.user_processes
    
    def get_system_whitelist(self) -> Set[str]:
        """
        获取系统白名单
        
        Returns:
            系统白名单进程集合
        """
        return self._config.system_processes.copy()
    
    def get_user_whitelist(self) -> Set[str]:
        """
        获取用户白名单
        
        Returns:
            用户白名单进程集合
        """
        return self._config.user_processes.copy()
    
    def get_patterns(self) -> List[str]:
        """
        获取所有正则表达式模式
        
        Returns:
            模式列表
        """
        return self._config.patterns.copy()
    
    def reset_to_defaults(self) -> None:
        """重置为默认配置"""
        self._config = WhitelistConfig(
            system_processes=self._get_default_system_whitelist(),
            user_processes=set(),
            patterns=self.DEFAULT_PATTERNS.copy(),
        )
        self._save_config(self._config)
        logger.info("Reset whitelist to defaults")
    
    def export_to_file(self, filepath: str, format: str = "yaml") -> bool:
        """
        导出白名单到文件
        
        Args:
            filepath: 文件路径
            format: 格式 (yaml 或 json)
            
        Returns:
            是否成功导出
        """
        try:
            data = {
                'system_processes': sorted(list(self._config.system_processes)),
                'user_processes': sorted(list(self._config.user_processes)),
                'patterns': self._config.patterns,
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                if format.lower() == 'json':
                    json.dump(data, f, indent=2, ensure_ascii=False)
                else:
                    yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"Exported whitelist to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export whitelist: {e}")
            return False
    
    def import_from_file(self, filepath: str, merge: bool = False) -> bool:
        """
        从文件导入白名单
        
        Args:
            filepath: 文件路径
            merge: 是否合并到现有配置，False则替换
            
        Returns:
            是否成功导入
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                if filepath.endswith('.json'):
                    data = json.load(f)
                else:
                    data = yaml.safe_load(f)
            
            if merge:
                self._config.system_processes.update(data.get('system_processes', []))
                self._config.user_processes.update(data.get('user_processes', []))
                self._config.patterns.extend(data.get('patterns', []))
                # 去重
                self._config.patterns = list(set(self._config.patterns))
            else:
                self._config.system_processes = set(data.get('system_processes', []))
                self._config.user_processes = set(data.get('user_processes', []))
                self._config.patterns = data.get('patterns', [])
            
            self._save_config(self._config)
            logger.info(f"Imported whitelist from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import whitelist: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取白名单统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "system_processes_count": len(self._config.system_processes),
            "user_processes_count": len(self._config.user_processes),
            "patterns_count": len(self._config.patterns),
            "config_path": self._config_path,
            "last_updated": self._config.last_updated,
        }
    
    def save(self) -> None:
        """手动保存配置"""
        self._save_config(self._config)


# 全局实例
_whitelist_manager: Optional[WhitelistManager] = None


def get_whitelist_manager(config_path: Optional[str] = None) -> WhitelistManager:
    """
    获取全局白名单管理器实例
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        WhitelistManager实例
    """
    global _whitelist_manager
    if _whitelist_manager is None:
        _whitelist_manager = WhitelistManager(config_path)
    return _whitelist_manager


def reset_whitelist_manager() -> None:
    """重置全局白名单管理器实例"""
    global _whitelist_manager
    _whitelist_manager = None
