"""
配置管理模块

提供配置加载、验证、动态更新和持久化功能。
支持YAML配置文件和环境变量覆盖。
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field, asdict
from datetime import datetime
import copy

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logging.warning("PyYAML not installed. Using JSON fallback for config.")

logger = logging.getLogger(__name__)


@dataclass
class SystemConfig:
    """系统配置"""
    name: str = "Building Energy Management System"
    version: str = "0.1.0"
    log_level: str = "INFO"
    data_dir: str = "./data"
    output_dir: str = "./output"


@dataclass
class SimulationConfig:
    """仿真配置"""
    idf_path: str = "models/building.idf"
    weather_path: str = "weather/CHN_Beijing.Beijing.545110_CSWD.epw"
    output_dir: str = "output/simulation"
    timestep: int = 15
    simulation_days: int = 1
    verbose: bool = False


@dataclass
class HVACConfig:
    """HVAC配置"""
    control_mode: str = "temperature"
    min_setpoint: float = 18.0
    max_setpoint: float = 26.0
    target_temperature: float = 22.0
    comfort_tolerance: float = 1.0
    hvac_power: float = 5.0
    natural_ventilation: bool = True


@dataclass
class RLConfig:
    """强化学习配置"""
    algorithm: str = "SAC"
    total_timesteps: int = 100000
    learning_rate: float = 0.0003
    batch_size: int = 256
    buffer_size: int = 100000
    gamma: float = 0.99
    log_dir: str = "output/rl_logs"
    save_dir: str = "output/models"
    eval_freq: int = 5000
    n_envs: int = 4


@dataclass
class RewardConfig:
    """奖励函数配置"""
    comfort_weight: float = 1.0
    energy_weight: float = 0.1
    temp_penalty_coeff: float = 2.0
    max_reward: float = 0.0


@dataclass
class WeatherConfig:
    """天气API配置"""
    provider: str = "openweathermap"
    api_key: str = ""
    cache_dir: str = "data/weather_cache"
    cache_ttl: int = 1
    default_lat: float = 39.9042
    default_lon: float = 116.4074


@dataclass
class AnomalyDetectionConfig:
    """异常检测配置"""
    algorithm: str = "isolation_forest"
    contamination: float = 0.1
    training_window: int = 168
    detection_interval: int = 15
    alert_threshold: float = 0.7


@dataclass
class RAGConfig:
    """知识库RAG配置"""
    knowledge_base_dir: str = "data/knowledge_base"
    graphrag_root_dir: str = "data/graphrag"
    graphrag_search_method: str = "hybrid"
    graphrag_community_level: int = 2
    vector_db_type: str = "chroma"
    vector_db_persist_dir: str = "data/vector_db"


@dataclass
class ControllerConfig:
    """主控制器配置"""
    # 运行模式: "simulation", "realtime", "hybrid"
    run_mode: str = "simulation"
    # 控制周期（秒）
    control_interval: int = 60
    # 数据收集间隔（秒）
    data_collection_interval: int = 15
    # 预测_horizon（小时）
    prediction_horizon: int = 24
    # 是否启用异常检测
    enable_anomaly_detection: bool = True
    # 是否启用知识库
    enable_knowledge_base: bool = True
    # 是否启用预测
    enable_prediction: bool = True
    # 告警通知方式: "log", "email", "webhook", "all"
    alert_notification: str = "log"
    # 自动保存间隔（分钟）
    autosave_interval: int = 30


class ConfigManager:
    """
    配置管理器
    
    负责加载、验证、管理和持久化系统配置。
    支持从YAML文件加载配置，并可通过环境变量覆盖。
    
    Attributes:
        config_path: 配置文件路径
        config_data: 原始配置数据字典
        system: 系统配置
        simulation: 仿真配置
        hvac: HVAC配置
        rl: 强化学习配置
        reward: 奖励函数配置
        weather: 天气配置
        anomaly_detection: 异常检测配置
        rag: 知识库配置
        controller: 主控制器配置
    
    Example:
        >>> config = ConfigManager("config.yaml")
        >>> config.load()
        >>> print(config.system.log_level)
        >>> config.system.log_level = "DEBUG"
        >>> config.save()
    """
    
    # 环境变量前缀
    ENV_PREFIX = "BEEMS_"
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，默认为 None（使用默认配置）
        """
        self.config_path = config_path
        self.config_data: Dict[str, Any] = {}
        
        # 初始化各配置段
        self.system = SystemConfig()
        self.simulation = SimulationConfig()
        self.hvac = HVACConfig()
        self.rl = RLConfig()
        self.reward = RewardConfig()
        self.weather = WeatherConfig()
        self.anomaly_detection = AnomalyDetectionConfig()
        self.rag = RAGConfig()
        self.controller = ControllerConfig()
        
        # 变更回调
        self._change_callbacks: List[callable] = []
        
        logger.info("ConfigManager initialized")
    
    def load(self, config_path: Optional[str] = None) -> None:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径，覆盖初始化时传入的路径
        
        Raises:
            FileNotFoundError: 当配置文件不存在时
            ValueError: 当配置文件格式不正确时
        """
        path = config_path or self.config_path
        
        if path is None:
            logger.info("No config path provided, using default configuration")
            self._apply_env_overrides()
            return
        
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        logger.info(f"Loading configuration from {path}")
        
        # 根据文件扩展名选择解析器
        if path.suffix in ['.yaml', '.yml']:
            if not YAML_AVAILABLE:
                raise ImportError("PyYAML required for YAML config files")
            with open(path, 'r', encoding='utf-8') as f:
                self.config_data = yaml.safe_load(f) or {}
        elif path.suffix == '.json':
            import json
            with open(path, 'r', encoding='utf-8') as f:
                self.config_data = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")
        
        # 解析配置到各配置段
        self._parse_config()
        
        # 应用环境变量覆盖
        self._apply_env_overrides()
        
        logger.info("Configuration loaded successfully")
    
    def _parse_config(self) -> None:
        """解析配置数据到各配置段"""
        # System
        if 'system' in self.config_data:
            self._update_dataclass_from_dict(
                self.system, self.config_data['system']
            )
        
        # Simulation
        if 'simulation' in self.config_data:
            self._update_dataclass_from_dict(
                self.simulation, self.config_data['simulation']
            )
        
        # HVAC
        if 'hvac' in self.config_data:
            self._update_dataclass_from_dict(
                self.hvac, self.config_data['hvac']
            )
        
        # RL
        if 'rl' in self.config_data:
            self._update_dataclass_from_dict(
                self.rl, self.config_data['rl']
            )
        
        # Reward
        if 'reward' in self.config_data:
            self._update_dataclass_from_dict(
                self.reward, self.config_data['reward']
            )
        
        # Weather
        if 'weather' in self.config_data:
            self._update_dataclass_from_dict(
                self.weather, self.config_data['weather']
            )
        
        # Anomaly Detection
        if 'anomaly_detection' in self.config_data:
            self._update_dataclass_from_dict(
                self.anomaly_detection, self.config_data['anomaly_detection']
            )
        
        # RAG
        if 'rag' in self.config_data:
            rag_data = self.config_data['rag']
            if 'graphrag' in rag_data:
                graphrag = rag_data['graphrag']
                if 'root_dir' in graphrag:
                    self.rag.graphrag_root_dir = graphrag['root_dir']
                if 'search_method' in graphrag:
                    self.rag.graphrag_search_method = graphrag['search_method']
                if 'community_level' in graphrag:
                    self.rag.graphrag_community_level = graphrag['community_level']
            if 'vector_db' in rag_data:
                vector_db = rag_data['vector_db']
                if 'type' in vector_db:
                    self.rag.vector_db_type = vector_db['type']
                if 'persist_dir' in vector_db:
                    self.rag.vector_db_persist_dir = vector_db['persist_dir']
            if 'knowledge_base_dir' in rag_data:
                self.rag.knowledge_base_dir = rag_data['knowledge_base_dir']
        
        # Controller (可能位于顶层或controller段)
        controller_data = self.config_data.get('controller', {})
        self._update_dataclass_from_dict(self.controller, controller_data)
    
    def _update_dataclass_from_dict(
        self, 
        dataclass_instance: Any, 
        data: Dict[str, Any]
    ) -> None:
        """
        从字典更新dataclass实例
        
        Args:
            dataclass_instance: dataclass实例
            data: 配置数据字典
        """
        for key, value in data.items():
            if hasattr(dataclass_instance, key):
                # 类型转换
                current_value = getattr(dataclass_instance, key)
                if isinstance(current_value, bool) and isinstance(value, str):
                    value = value.lower() in ('true', '1', 'yes', 'on')
                elif isinstance(current_value, int) and isinstance(value, str):
                    value = int(value)
                elif isinstance(current_value, float) and isinstance(value, str):
                    value = float(value)
                
                setattr(dataclass_instance, key, value)
    
    def _apply_env_overrides(self) -> None:
        """应用环境变量覆盖配置"""
        import os
        
        for key, value in os.environ.items():
            if key.startswith(self.ENV_PREFIX):
                # 解析环境变量名
                # 格式: BEEMS_SECTION_KEY 或 BEEMS_KEY
                parts = key[len(self.ENV_PREFIX):].lower().split('_')
                
                if len(parts) >= 2:
                    section = parts[0]
                    config_key = '_'.join(parts[1:])
                    
                    # 映射到配置段
                    section_map = {
                        'system': self.system,
                        'simulation': self.simulation,
                        'hvac': self.hvac,
                        'rl': self.rl,
                        'reward': self.reward,
                        'weather': self.weather,
                        'anomaly': self.anomaly_detection,
                        'rag': self.rag,
                        'controller': self.controller,
                    }
                    
                    if section in section_map:
                        instance = section_map[section]
                        if hasattr(instance, config_key):
                            # 类型转换
                            current_value = getattr(instance, config_key)
                            if isinstance(current_value, bool):
                                value = value.lower() in ('true', '1', 'yes', 'on')
                            elif isinstance(current_value, int):
                                value = int(value)
                            elif isinstance(current_value, float):
                                value = float(value)
                            
                            setattr(instance, config_key, value)
                            logger.debug(f"Override from env: {key}={value}")
    
    def save(self, config_path: Optional[str] = None) -> None:
        """
        保存配置到文件
        
        Args:
            config_path: 保存路径，默认为加载时的路径
        """
        path = config_path or self.config_path
        
        if path is None:
            raise ValueError("No config path specified for saving")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # 构建配置字典
        config_dict = {
            'system': asdict(self.system),
            'simulation': asdict(self.simulation),
            'hvac': asdict(self.hvac),
            'rl': asdict(self.rl),
            'reward': asdict(self.reward),
            'weather': asdict(self.weather),
            'anomaly_detection': asdict(self.anomaly_detection),
            'rag': {
                'knowledge_base_dir': self.rag.knowledge_base_dir,
                'graphrag': {
                    'root_dir': self.rag.graphrag_root_dir,
                    'search_method': self.rag.graphrag_search_method,
                    'community_level': self.rag.graphrag_community_level,
                },
                'vector_db': {
                    'type': self.rag.vector_db_type,
                    'persist_dir': self.rag.vector_db_persist_dir,
                }
            },
            'controller': asdict(self.controller),
        }
        
        # 添加保存时间戳
        config_dict['_meta'] = {
            'saved_at': datetime.now().isoformat(),
            'version': self.system.version,
        }
        
        # 保存
        if path.suffix in ['.yaml', '.yml']:
            if not YAML_AVAILABLE:
                raise ImportError("PyYAML required for YAML config files")
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        else:
            import json
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Configuration saved to {path}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值（支持点号路径）
        
        Args:
            key: 配置键，如 "system.log_level" 或 "hvac.target_temperature"
            default: 默认值
        
        Returns:
            配置值或默认值
        """
        parts = key.split('.')
        
        if len(parts) == 1:
            # 在controller中查找
            return getattr(self.controller, key, default)
        
        section_name = parts[0]
        attr_name = parts[1]
        
        section_map = {
            'system': self.system,
            'simulation': self.simulation,
            'hvac': self.hvac,
            'rl': self.rl,
            'reward': self.reward,
            'weather': self.weather,
            'anomaly_detection': self.anomaly_detection,
            'rag': self.rag,
            'controller': self.controller,
        }
        
        if section_name in section_map:
            section = section_map[section_name]
            return getattr(section, attr_name, default)
        
        return default
    
    def set(self, key: str, value: Any) -> None:
        """
        设置配置值（支持点号路径）
        
        Args:
            key: 配置键
            value: 配置值
        """
        parts = key.split('.')
        
        if len(parts) == 1:
            setattr(self.controller, key, value)
        else:
            section_name = parts[0]
            attr_name = parts[1]
            
            section_map = {
                'system': self.system,
                'simulation': self.simulation,
                'hvac': self.hvac,
                'rl': self.rl,
                'reward': self.reward,
                'weather': self.weather,
                'anomaly_detection': self.anomaly_detection,
                'rag': self.rag,
                'controller': self.controller,
            }
            
            if section_name in section_map:
                setattr(section_map[section_name], attr_name, value)
        
        # 触发变更回调
        self._notify_change(key, value)
    
    def register_change_callback(self, callback: callable) -> None:
        """
        注册配置变更回调
        
        Args:
            callback: 回调函数，接收(key, value)参数
        """
        self._change_callbacks.append(callback)
    
    def _notify_change(self, key: str, value: Any) -> None:
        """通知配置变更"""
        for callback in self._change_callbacks:
            try:
                callback(key, value)
            except Exception as e:
                logger.warning(f"Config change callback failed: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            完整配置字典
        """
        return {
            'system': asdict(self.system),
            'simulation': asdict(self.simulation),
            'hvac': asdict(self.hvac),
            'rl': asdict(self.rl),
            'reward': asdict(self.reward),
            'weather': asdict(self.weather),
            'anomaly_detection': asdict(self.anomaly_detection),
            'rag': asdict(self.rag),
            'controller': asdict(self.controller),
        }
    
    def validate(self) -> List[str]:
        """
        验证配置有效性
        
        Returns:
            错误信息列表，空列表表示验证通过
        """
        errors = []
        
        # 验证系统配置
        if self.system.log_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            errors.append(f"Invalid log_level: {self.system.log_level}")
        
        # 验证HVAC配置
        if self.hvac.min_setpoint >= self.hvac.max_setpoint:
            errors.append("HVAC min_setpoint must be less than max_setpoint")
        
        if not (16 <= self.hvac.target_temperature <= 30):
            errors.append("HVAC target_temperature should be between 16 and 30")
        
        # 验证RL配置
        if self.rl.algorithm not in ['SAC', 'PPO', 'TD3', 'A2C']:
            errors.append(f"Unsupported RL algorithm: {self.rl.algorithm}")
        
        # 验证异常检测配置
        if not (0 < self.anomaly_detection.contamination <= 0.5):
            errors.append("anomaly_detection.contamination must be in (0, 0.5]")
        
        # 验证控制器配置
        if self.controller.run_mode not in ['simulation', 'realtime', 'hybrid']:
            errors.append(f"Invalid run_mode: {self.controller.run_mode}")
        
        return errors
    
    def create_default_config(self, path: str) -> None:
        """
        创建默认配置文件
        
        Args:
            path: 配置文件保存路径
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        default_yaml = """# 建筑智能节能系统默认配置

# 系统配置
system:
  name: "Building Energy Management System"
  version: "0.1.0"
  log_level: "INFO"
  data_dir: "./data"
  output_dir: "./output"

# EnergyPlus建筑模拟配置
simulation:
  idf_path: "models/building.idf"
  weather_path: "weather/CHN_Beijing.Beijing.545110_CSWD.epw"
  output_dir: "output/simulation"
  timestep: 15
  simulation_days: 1
  verbose: false

# HVAC控制配置
hvac:
  control_mode: "temperature"
  min_setpoint: 18.0
  max_setpoint: 26.0
  target_temperature: 22.0
  comfort_tolerance: 1.0
  hvac_power: 5.0
  natural_ventilation: true

# 强化学习配置
rl:
  algorithm: "SAC"
  total_timesteps: 100000
  learning_rate: 0.0003
  batch_size: 256
  buffer_size: 100000
  gamma: 0.99
  log_dir: "output/rl_logs"
  save_dir: "output/models"
  eval_freq: 5000
  n_envs: 4

# 奖励函数配置
reward:
  comfort_weight: 1.0
  energy_weight: 0.1
  temp_penalty_coeff: 2.0
  max_reward: 0.0

# 天气API配置
weather:
  provider: "openweathermap"
  api_key: ""
  cache_dir: "data/weather_cache"
  cache_ttl: 1
  default_lat: 39.9042
  default_lon: 116.4074

# 异常检测配置
anomaly_detection:
  algorithm: "isolation_forest"
  contamination: 0.1
  training_window: 168
  detection_interval: 15
  alert_threshold: 0.7

# 知识库RAG配置
rag:
  knowledge_base_dir: "data/knowledge_base"
  graphrag:
    root_dir: "data/graphrag"
    search_method: "hybrid"
    community_level: 2
  vector_db:
    type: "chroma"
    persist_dir: "data/vector_db"

# 主控制器配置
controller:
  run_mode: "simulation"
  control_interval: 60
  data_collection_interval: 15
  prediction_horizon: 24
  enable_anomaly_detection: true
  enable_knowledge_base: true
  enable_prediction: true
  alert_notification: "log"
  autosave_interval: 30
"""
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(default_yaml)
        
        logger.info(f"Default configuration created at {path}")


# 全局配置实例
_config_instance: Optional[ConfigManager] = None


def get_config(config_path: Optional[str] = None) -> ConfigManager:
    """
    获取全局配置实例（单例模式）
    
    Args:
        config_path: 配置文件路径，首次调用时有效
    
    Returns:
        ConfigManager实例
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = ConfigManager(config_path)
        if config_path:
            _config_instance.load()
    
    return _config_instance


def reset_config() -> None:
    """重置全局配置实例"""
    global _config_instance
    _config_instance = None
