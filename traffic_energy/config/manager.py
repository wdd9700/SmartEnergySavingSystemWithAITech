#!/usr/bin/env python3
"""
配置管理模块

提供统一的配置加载、验证和管理功能。
支持YAML配置文件和环境变量覆盖。
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """模型配置"""
    name: str = "yolo12n.pt"
    conf_threshold: float = 0.5
    iou_threshold: float = 0.45
    device: str = "auto"
    classes: List[int] = field(default_factory=lambda: [2, 3, 5, 7])


@dataclass
class TrackerConfig:
    """跟踪器配置"""
    type: str = "botsort"
    track_high_thresh: float = 0.6
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.7
    track_buffer: int = 60
    match_thresh: float = 0.8
    proximity_thresh: float = 0.5
    appearance_thresh: float = 0.25
    cmc_method: str = "ecc"
    frame_rate: int = 30
    lambda_: float = 0.985


@dataclass
class DetectionConfig:
    """检测配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)


@dataclass
class ReidConfig:
    """重识别配置"""
    model_name: str = "veriwild_bagtricks_R50-ibn"
    model_path: str = "models/fastreid/veriwild_bagtricks_R50-ibn.pth"
    device: str = "auto"
    input_size: List[int] = field(default_factory=lambda: [128, 256])
    feature_dim: int = 2048
    similarity_threshold: float = 0.7
    top_k: int = 10


@dataclass
class CameraNode:
    """摄像头节点配置"""
    id: str
    name: str
    location: Dict[str, float]
    stream_url: str
    direction: int
    roi: List[List[int]]
    calibration_points: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class CameraEdge:
    """摄像头连接关系"""
    from_id: str
    to_id: str
    distance: float
    min_travel_time: float
    max_travel_time: float
    direction: str


@dataclass
class CameraTopology:
    """摄像头拓扑配置"""
    nodes: List[CameraNode] = field(default_factory=list)
    edges: List[CameraEdge] = field(default_factory=list)


@dataclass
class TrafficConfig:
    """交通系统完整配置"""
    system_name: str = "Smart Traffic Energy System"
    version: str = "1.0.0"
    log_level: str = "INFO"
    log_dir: str = "logs/traffic_energy"
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    reid: ReidConfig = field(default_factory=ReidConfig)
    camera_topology: CameraTopology = field(default_factory=CameraTopology)


class ConfigManager:
    """配置管理器
    
    负责加载、验证和管理系统配置。
    支持从YAML文件加载，并可通过环境变量覆盖。
    
    Attributes:
        config_path: 配置文件路径
        config: 当前配置对象
        
    Example:
        >>> manager = ConfigManager("config/default_config.yaml")
        >>> manager.load()
        >>> print(manager.config.system_name)
        "Smart Traffic Energy System"
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化配置管理器
        
        Args:
            config_path: 配置文件路径，默认为None
        """
        self.config_path = config_path
        self._config: Optional[TrafficConfig] = None
        self._raw_config: Dict[str, Any] = {}
        
    def load(self, config_path: Optional[str] = None) -> TrafficConfig:
        """加载配置文件
        
        Args:
            config_path: 配置文件路径，覆盖初始化时的路径
            
        Returns:
            TrafficConfig: 加载的配置对象
            
        Raises:
            FileNotFoundError: 配置文件不存在
            yaml.YAMLError: YAML解析错误
        """
        path = config_path or self.config_path
        if not path:
            # 使用默认配置
            self._config = TrafficConfig()
            return self._config
            
        config_file = Path(path)
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")
            
        with open(config_file, 'r', encoding='utf-8') as f:
            self._raw_config = yaml.safe_load(f) or {}
            
        self._config = self._parse_config(self._raw_config)
        
        # 应用环境变量覆盖
        self._apply_env_overrides()
        
        return self._config
    
    def _parse_config(self, raw: Dict[str, Any]) -> TrafficConfig:
        """解析原始配置字典
        
        Args:
            raw: 原始配置字典
            
        Returns:
            TrafficConfig: 解析后的配置对象
        """
        config = TrafficConfig()
        
        # 解析各模块配置
        config = self._parse_system_config(config, raw)
        config = self._parse_detection_config(config, raw)
        config = self._parse_reid_config(config, raw)
        
        return config
    
    def _parse_system_config(self, config: TrafficConfig, raw: Dict[str, Any]) -> TrafficConfig:
        """解析系统配置
        
        Args:
            config: 当前配置对象
            raw: 原始配置字典
            
        Returns:
            更新后的配置对象
        """
        if 'system' not in raw:
            return config
            
        sys_cfg = raw['system']
        config.system_name = sys_cfg.get('name', config.system_name)
        config.version = sys_cfg.get('version', config.version)
        config.log_level = sys_cfg.get('log_level', config.log_level)
        config.log_dir = sys_cfg.get('log_dir', config.log_dir)
        
        return config
    
    def _parse_detection_config(self, config: TrafficConfig, raw: Dict[str, Any]) -> TrafficConfig:
        """解析检测配置
        
        Args:
            config: 当前配置对象
            raw: 原始配置字典
            
        Returns:
            更新后的配置对象
        """
        if 'detection' not in raw:
            return config
            
        det_cfg = raw['detection']
        
        # 解析模型配置
        if 'model' in det_cfg:
            config.detection.model = self._create_model_config(det_cfg['model'])
        
        # 解析跟踪器配置
        if 'tracker' in det_cfg:
            config.detection.tracker = self._create_tracker_config(det_cfg['tracker'])
        
        return config
    
    def _create_model_config(self, model_dict: Dict[str, Any]) -> ModelConfig:
        """创建模型配置对象
        
        Args:
            model_dict: 模型配置字典
            
        Returns:
            ModelConfig对象
        """
        return ModelConfig(
            name=model_dict.get('name', 'yolo12n.pt'),
            conf_threshold=model_dict.get('conf_threshold', 0.5),
            iou_threshold=model_dict.get('iou_threshold', 0.45),
            device=model_dict.get('device', 'auto'),
            classes=model_dict.get('classes', [2, 3, 5, 7])
        )
    
    def _create_tracker_config(self, tracker_dict: Dict[str, Any]) -> TrackerConfig:
        """创建跟踪器配置对象
        
        Args:
            tracker_dict: 跟踪器配置字典
            
        Returns:
            TrackerConfig对象
        """
        return TrackerConfig(
            type=tracker_dict.get('type', 'botsort'),
            track_high_thresh=tracker_dict.get('track_high_thresh', 0.6),
            track_low_thresh=tracker_dict.get('track_low_thresh', 0.1),
            new_track_thresh=tracker_dict.get('new_track_thresh', 0.7),
            track_buffer=tracker_dict.get('track_buffer', 60),
            match_thresh=tracker_dict.get('match_thresh', 0.8),
            proximity_thresh=tracker_dict.get('proximity_thresh', 0.5),
            appearance_thresh=tracker_dict.get('appearance_thresh', 0.25),
            cmc_method=tracker_dict.get('cmc_method', 'ecc'),
            frame_rate=tracker_dict.get('frame_rate', 30),
            lambda_=tracker_dict.get('lambda_', 0.985)
        )
    
    def _parse_reid_config(self, config: TrafficConfig, raw: Dict[str, Any]) -> TrafficConfig:
        """解析ReID配置
        
        Args:
            config: 当前配置对象
            raw: 原始配置字典
            
        Returns:
            更新后的配置对象
        """
        if 'reid' not in raw:
            return config
            
        reid_cfg = raw['reid']
        
        # 解析模型配置
        if 'model' in reid_cfg:
            config.reid = self._create_reid_model_config(reid_cfg['model'])
        
        # 解析匹配配置
        if 'matching' in reid_cfg and config.reid:
            config.reid = self._apply_reid_matching_config(
                config.reid, reid_cfg['matching']
            )
        
        return config
    
    def _create_reid_model_config(self, model_dict: Dict[str, Any]) -> ReidConfig:
        """创建ReID模型配置对象
        
        Args:
            model_dict: ReID模型配置字典
            
        Returns:
            ReidConfig对象
        """
        return ReidConfig(
            model_name=model_dict.get('name', 'veriwild_bagtricks_R50-ibn'),
            model_path=model_dict.get('path', ''),
            device=model_dict.get('device', 'auto'),
            input_size=model_dict.get('input_size', [128, 256]),
            feature_dim=model_dict.get('feature_dim', 2048)
        )
    
    def _apply_reid_matching_config(self, config: ReidConfig, 
                                     matching_dict: Dict[str, Any]) -> ReidConfig:
        """应用ReID匹配配置
        
        Args:
            config: 当前ReID配置
            matching_dict: 匹配配置字典
            
        Returns:
            更新后的ReID配置
        """
        config.similarity_threshold = matching_dict.get('similarity_threshold', 0.7)
        config.top_k = matching_dict.get('top_k', 10)
        return config
    
    def _apply_env_overrides(self) -> None:
        """应用环境变量覆盖配置"""
        if self._config is None:
            return
            
        # 日志级别
        log_level = os.getenv('TRAFFIC_LOG_LEVEL')
        if log_level:
            self._config.log_level = log_level
            
        # 设备配置
        device = os.getenv('TRAFFIC_DEVICE')
        if device:
            self._config.detection.model.device = device
            self._config.reid.device = device
            
        # 模型路径
        model_path = os.getenv('TRAFFIC_MODEL_PATH')
        if model_path:
            self._config.detection.model.name = model_path
    
    @property
    def config(self) -> TrafficConfig:
        """获取当前配置
        
        Returns:
            TrafficConfig: 当前配置对象
            
        Raises:
            RuntimeError: 配置尚未加载
        """
        if self._config is None:
            raise RuntimeError("配置尚未加载，请先调用load()")
        return self._config
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项
        
        支持点号分隔的嵌套键，如 "detection.model.name"
        
        Args:
            key: 配置键
            default: 默认值
            
        Returns:
            配置值或默认值
        """
        keys = key.split('.')
        value = self._raw_config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def reload(self) -> TrafficConfig:
        """重新加载配置
        
        Returns:
            TrafficConfig: 重新加载后的配置
        """
        return self.load(self.config_path)
    
    def save(self, config_path: Optional[str] = None) -> None:
        """保存配置到文件
        
        Args:
            config_path: 保存路径，默认为原路径
        """
        path = config_path or self.config_path
        if not path:
            raise ValueError("未指定保存路径")
            
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self._raw_config, f, default_flow_style=False, allow_unicode=True)


# 全局配置管理器实例
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """获取全局配置管理器实例
    
    Returns:
        ConfigManager: 全局配置管理器
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def load_config(config_path: str) -> TrafficConfig:
    """便捷函数：加载配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        TrafficConfig: 加载的配置
    """
    manager = get_config_manager()
    return manager.load(config_path)
