"""
建筑智能节能系统 - 主控制程序

实现系统的核心控制逻辑，协调各模块协同工作：
- 异常检测模块
- 知识库模块
- 预测模型模块

提供系统初始化、运行、监控和关闭的完整生命周期管理。
"""

import os
import sys
import time
import logging
import signal
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import copy

import numpy as np
import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入配置管理
from building_energy.config.manager import ConfigManager, get_config

# 导入核心模块
try:
    from building_energy.models.anomaly_detector import AnomalyDetector, AnomalyAlert
    ANOMALY_AVAILABLE = True
except ImportError as e:
    ANOMALY_AVAILABLE = False
    logging.warning(f"AnomalyDetector not available: {e}")
    AnomalyAlert = None

try:
    from building_energy.knowledge.graph_rag import KnowledgeBase, QueryResult
    KNOWLEDGE_AVAILABLE = True
except ImportError as e:
    KNOWLEDGE_AVAILABLE = False
    logging.warning(f"KnowledgeBase not available: {e}")
    QueryResult = None

try:
    from building_energy.models.predictor import EnergyPredictor, PredictionResult
    PREDICTOR_AVAILABLE = True
except ImportError as e:
    PREDICTOR_AVAILABLE = False
    logging.warning(f"EnergyPredictor not available: {e}")
    PredictionResult = None

# 导入HVAC环境（如果可用）
try:
    from building_energy.env.hvac_env import HVACEnv
    HVAC_ENV_AVAILABLE = True
except ImportError as e:
    HVAC_ENV_AVAILABLE = False
    logging.warning(f"HVACEnv not available: {e}")

# 导入建筑模拟器（如果可用）
try:
    from building_energy.core.building_simulator import BuildingSimulator
    SIMULATOR_AVAILABLE = True
except ImportError as e:
    SIMULATOR_AVAILABLE = False
    logging.warning(f"BuildingSimulator not available: {e}")

logger = logging.getLogger(__name__)


class SystemState(Enum):
    """系统状态枚举"""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    SHUTTING_DOWN = "shutting_down"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class SystemStatus:
    """系统状态数据结构"""
    state: SystemState = SystemState.INITIALIZING
    start_time: Optional[datetime] = None
    uptime_seconds: float = 0.0
    current_cycle: int = 0
    total_cycles: int = 0
    last_error: Optional[str] = None
    active_alerts: int = 0
    predictions_made: int = 0
    energy_saved_estimate: float = 0.0  # kWh
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'state': self.state.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'uptime_seconds': self.uptime_seconds,
            'current_cycle': self.current_cycle,
            'total_cycles': self.total_cycles,
            'last_error': self.last_error,
            'active_alerts': self.active_alerts,
            'predictions_made': self.predictions_made,
            'energy_saved_estimate': self.energy_saved_estimate,
        }


@dataclass
class ControlDecision:
    """控制决策数据结构"""
    timestamp: datetime
    action: str
    target_temperature: Optional[float] = None
    hvac_power: Optional[float] = None
    reason: str = ""
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'action': self.action,
            'target_temperature': self.target_temperature,
            'hvac_power': self.hvac_power,
            'reason': self.reason,
            'confidence': self.confidence,
        }


class BuildingController:
    """
    建筑智能节能系统主控制器
    
    协调异常检测、知识库、预测模型等模块，实现智能建筑能耗管理。
    
    Attributes:
        config: 配置管理器
        state: 当前系统状态
        status: 系统状态信息
        anomaly_detector: 异常检测器
        knowledge_base: 知识库
        energy_predictor: 能耗预测器
    
    Example:
        >>> controller = BuildingController("config.yaml")
        >>> controller.initialize()
        >>> controller.run()
        >>> # ... 运行一段时间后 ...
        >>> controller.shutdown()
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化建筑控制器
        
        Args:
            config_path: 配置文件路径，None则使用默认配置
        """
        self.config_path = config_path
        self.config: ConfigManager = get_config(config_path)
        
        # 系统状态
        self.state = SystemState.INITIALIZING
        self.status = SystemStatus()
        
        # 模块实例
        self.anomaly_detector: Optional[AnomalyDetector] = None
        self.knowledge_base: Optional[KnowledgeBase] = None
        self.energy_predictor: Optional[EnergyPredictor] = None
        self.hvac_env: Optional[Any] = None
        self.simulator: Optional[Any] = None
        
        # 运行控制
        self._running = False
        self._paused = False
        self._shutdown_event = threading.Event()
        self._control_thread: Optional[threading.Thread] = None
        
        # 数据存储
        self._sensor_data_buffer: List[Dict[str, Any]] = []
        self._control_history: List[ControlDecision] = []
        self._alert_history: List[Any] = []
        
        # 回调函数
        self._cycle_callbacks: List[Callable] = []
        self._alert_callbacks: List[Callable] = []
        
        # 自动保存
        self._last_autosave = datetime.now()
        
        # 信号处理
        self._setup_signal_handlers()
        
        logger.info("BuildingController initialized")
    
    def _setup_signal_handlers(self) -> None:
        """设置信号处理器"""
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except (ValueError, OSError):
            # Windows或信号已注册时可能失败
            pass
    
    def _signal_handler(self, signum, frame) -> None:
        """信号处理器"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown()
    
    def initialize(self) -> None:
        """
        初始化系统
        
        加载配置、初始化各模块、验证系统状态。
        
        Raises:
            RuntimeError: 当初始化失败时
        """
        logger.info("Initializing Building Energy Management System...")
        self.state = SystemState.INITIALIZING
        
        try:
            # 加载配置
            if self.config_path:
                self.config.load(self.config_path)
            
            # 验证配置
            errors = self.config.validate()
            if errors:
                raise RuntimeError(f"Configuration validation failed: {errors}")
            
            # 设置日志
            self._setup_logging()
            
            # 创建必要的目录
            self._create_directories()
            
            # 初始化异常检测模块
            if self.config.controller.enable_anomaly_detection and ANOMALY_AVAILABLE:
                self._init_anomaly_detector()
            
            # 初始化知识库模块
            if self.config.controller.enable_knowledge_base and KNOWLEDGE_AVAILABLE:
                self._init_knowledge_base()
            
            # 初始化预测模型
            if self.config.controller.enable_prediction and PREDICTOR_AVAILABLE:
                self._init_predictor()
            
            # 初始化HVAC环境或模拟器
            self._init_environment()
            
            # 更新状态
            self.state = SystemState.READY
            self.status.start_time = datetime.now()
            
            logger.info("System initialization completed successfully")
            
        except Exception as e:
            self.state = SystemState.ERROR
            self.status.last_error = str(e)
            logger.error(f"System initialization failed: {e}")
            raise RuntimeError(f"Initialization failed: {e}")
    
    def _setup_logging(self) -> None:
        """设置日志系统"""
        log_level = getattr(logging, self.config.system.log_level.upper(), logging.INFO)
        
        # 创建日志目录
        log_dir = Path(self.config.system.output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 配置根日志记录器
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(
                    log_dir / f"building_energy_{datetime.now().strftime('%Y%m%d')}.log",
                    encoding='utf-8'
                )
            ]
        )
    
    def _create_directories(self) -> None:
        """创建必要的目录结构"""
        dirs_to_create = [
            self.config.system.data_dir,
            self.config.system.output_dir,
            self.config.simulation.output_dir,
            self.config.rl.log_dir,
            self.config.rl.save_dir,
            self.config.weather.cache_dir,
            self.config.rag.knowledge_base_dir,
            self.config.rag.graphrag_root_dir,
            self.config.rag.vector_db_persist_dir,
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def _init_anomaly_detector(self) -> None:
        """初始化异常检测器"""
        logger.info("Initializing anomaly detector...")
        
        ad_config = self.config.anomaly_detection
        
        # 映射算法名称
        algorithm_map = {
            'isolation_forest': 'iforest',
            'autoencoder': 'autoencoder',
            'lof': 'lof',
            'hbos': 'hbos',
        }
        
        algorithm = algorithm_map.get(ad_config.algorithm, 'iforest')
        
        self.anomaly_detector = AnomalyDetector(
            algorithm=algorithm,
            contamination=ad_config.contamination,
            alert_threshold=ad_config.alert_threshold,
            feature_names=[
                'outdoor_temp', 'indoor_temp', 'indoor_humidity',
                'hvac_power', 'occupancy', 'co2_level'
            ]
        )
        
        # 尝试加载已有模型
        model_path = Path(self.config.rl.save_dir) / "anomaly_detector.pkl"
        if model_path.exists():
            try:
                self.anomaly_detector.load(str(model_path))
                logger.info(f"Loaded anomaly detector model from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load anomaly detector model: {e}")
        
        logger.info("Anomaly detector initialized")
    
    def _init_knowledge_base(self) -> None:
        """初始化知识库"""
        logger.info("Initializing knowledge base...")
        
        self.knowledge_base = KnowledgeBase(
            root_dir=self.config.rag.graphrag_root_dir
        )
        
        # 如果知识库未索引且有文档，则执行索引
        if not self.knowledge_base._indexed:
            kb_dir = Path(self.config.rag.knowledge_base_dir)
            if kb_dir.exists():
                doc_files = list(kb_dir.glob("*.md")) + list(kb_dir.glob("*.txt"))
                if doc_files:
                    logger.info(f"Found {len(doc_files)} documents to index")
                    for doc_file in doc_files:
                        self.knowledge_base.add_document(str(doc_file))
                    self.knowledge_base.index()
        
        logger.info("Knowledge base initialized")
    
    def _init_predictor(self) -> None:
        """初始化能耗预测器"""
        logger.info("Initializing energy predictor...")
        
        self.energy_predictor = EnergyPredictor(
            model_type="lstm",
            seq_length=24,
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
            learning_rate=0.001
        )
        
        # 尝试加载已有模型
        model_path = Path(self.config.rl.save_dir) / "energy_predictor.pt"
        if model_path.exists():
            try:
                self.energy_predictor.load(str(model_path))
                logger.info(f"Loaded energy predictor model from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load energy predictor model: {e}")
        
        logger.info("Energy predictor initialized")
    
    def _init_environment(self) -> None:
        """初始化环境（模拟器或真实环境）"""
        run_mode = self.config.controller.run_mode
        
        if run_mode == "simulation" and SIMULATOR_AVAILABLE:
            logger.info("Initializing building simulator...")
            self.simulator = BuildingSimulator(
                idf_path=self.config.simulation.idf_path,
                weather_path=self.config.simulation.weather_path
            )
        elif run_mode in ["realtime", "hybrid"] and HVAC_ENV_AVAILABLE:
            logger.info("Initializing HVAC environment...")
            self.hvac_env = HVACEnv(
                config=self.config.hvac.__dict__
            )
        else:
            logger.warning(f"Running in {run_mode} mode without environment backend")
    
    def run(self, duration_seconds: Optional[int] = None) -> None:
        """
        运行主控制循环
        
        Args:
            duration_seconds: 运行持续时间（秒），None表示一直运行
        
        Raises:
            RuntimeError: 当系统未初始化或已在运行时
        """
        if self.state not in [SystemState.READY, SystemState.PAUSED]:
            raise RuntimeError(f"Cannot run in state: {self.state}")
        
        if self._running:
            raise RuntimeError("Controller is already running")
        
        logger.info("Starting main control loop...")
        self._running = True
        self._paused = False
        self._shutdown_event.clear()
        self.state = SystemState.RUNNING
        
        start_time = time.time()
        
        try:
            while self._running and not self._shutdown_event.is_set():
                if self._paused:
                    time.sleep(0.1)
                    continue
                
                cycle_start = time.time()
                
                # 执行控制周期
                self._execute_control_cycle()
                
                # 更新状态
                self.status.current_cycle += 1
                self.status.total_cycles += 1
                self.status.uptime_seconds = time.time() - (self.status.start_time or datetime.now()).timestamp()
                
                # 检查持续时间
                if duration_seconds and (time.time() - start_time) >= duration_seconds:
                    logger.info(f"Reached duration limit: {duration_seconds}s")
                    break
                
                # 计算睡眠时间来维持控制周期
                cycle_duration = time.time() - cycle_start
                sleep_time = max(0, self.config.controller.control_interval - cycle_duration)
                
                if sleep_time > 0:
                    # 使用事件等待以便可以中断
                    self._shutdown_event.wait(timeout=sleep_time)
        
        except Exception as e:
            self.state = SystemState.ERROR
            self.status.last_error = str(e)
            logger.error(f"Control loop error: {e}", exc_info=True)
            raise
        
        finally:
            self._running = False
            if self.state != SystemState.ERROR:
                self.state = SystemState.READY
            
            logger.info("Main control loop stopped")
    
    def _execute_control_cycle(self) -> None:
        """执行单个控制周期"""
        try:
            # 1. 收集传感器数据
            sensor_data = self._collect_sensor_data()
            
            # 2. 异常检测
            if self.anomaly_detector and self.config.controller.enable_anomaly_detection:
                self._perform_anomaly_detection(sensor_data)
            
            # 3. 能耗预测
            prediction = None
            if self.energy_predictor and self.config.controller.enable_prediction:
                prediction = self._perform_prediction(sensor_data)
            
            # 4. 查询知识库获取优化建议
            advice = None
            if self.knowledge_base and self.config.controller.enable_knowledge_base:
                advice = self._get_optimization_advice(sensor_data)
            
            # 5. 生成控制决策
            decision = self._make_control_decision(sensor_data, prediction, advice)
            
            # 6. 执行控制动作
            self._execute_control_action(decision)
            
            # 7. 数据记录
            self._record_data(sensor_data, decision, prediction)
            
            # 8. 自动保存
            self._check_autosave()
            
            # 触发周期回调
            self._trigger_cycle_callbacks(sensor_data, decision)
            
        except Exception as e:
            logger.error(f"Control cycle error: {e}", exc_info=True)
            self.status.last_error = str(e)
    
    def _collect_sensor_data(self) -> Dict[str, Any]:
        """收集传感器数据"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'outdoor_temp': 25.0,  # 默认值
            'indoor_temp': 24.0,
            'indoor_humidity': 50.0,
            'hvac_power': 2.5,
            'occupancy': 0,
            'co2_level': 400.0,
        }
        
        # 从模拟器或真实环境获取数据
        if self.simulator:
            # 从模拟器获取数据
            try:
                sim_data = self.simulator.get_sensor_data()
                data.update(sim_data)
            except Exception as e:
                logger.warning(f"Failed to get simulator data: {e}")
        
        elif self.hvac_env:
            # 从HVAC环境获取数据
            try:
                env_data = self.hvac_env.get_observation()
                data.update(env_data)
            except Exception as e:
                logger.warning(f"Failed to get HVAC environment data: {e}")
        
        # 添加到缓冲区
        self._sensor_data_buffer.append(data)
        
        # 限制缓冲区大小
        max_buffer_size = self.config.anomaly_detection.training_window
        if len(self._sensor_data_buffer) > max_buffer_size:
            self._sensor_data_buffer = self._sensor_data_buffer[-max_buffer_size:]
        
        return data
    
    def _perform_anomaly_detection(self, sensor_data: Dict[str, Any]) -> None:
        """执行异常检测"""
        if not self.anomaly_detector or not self.anomaly_detector.is_fitted:
            # 尝试训练模型
            if len(self._sensor_data_buffer) >= 10:
                self._train_anomaly_detector()
            return
        
        # 准备特征
        features = self._extract_features(sensor_data)
        X = np.array([features])
        
        # 预测
        try:
            predictions = self.anomaly_detector.predict(X)
            
            if predictions[0] == 1:  # 异常
                alerts = self.anomaly_detector.get_alerts(since=datetime.now() - timedelta(minutes=1))
                if alerts:
                    self.status.active_alerts = len(self.anomaly_detector.get_alerts())
                    self._alert_history.extend(alerts)
                    
                    # 触发告警回调
                    for alert in alerts:
                        self._trigger_alert_callbacks(alert)
                    
                    logger.warning(f"Anomaly detected: {alerts[-1].description}")
        
        except Exception as e:
            logger.warning(f"Anomaly detection failed: {e}")
    
    def _train_anomaly_detector(self) -> None:
        """训练异常检测模型"""
        if len(self._sensor_data_buffer) < 10:
            return
        
        logger.info(f"Training anomaly detector with {len(self._sensor_data_buffer)} samples...")
        
        try:
            # 准备训练数据
            features_list = []
            for data in self._sensor_data_buffer:
                features = self._extract_features(data)
                features_list.append(features)
            
            X = np.array(features_list)
            
            # 训练
            self.anomaly_detector.fit(X)
            
            logger.info("Anomaly detector training completed")
        
        except Exception as e:
            logger.warning(f"Anomaly detector training failed: {e}")
    
    def _extract_features(self, data: Dict[str, Any]) -> List[float]:
        """从传感器数据提取特征"""
        return [
            float(data.get('outdoor_temp', 25.0)),
            float(data.get('indoor_temp', 24.0)),
            float(data.get('indoor_humidity', 50.0)),
            float(data.get('hvac_power', 2.5)),
            float(data.get('occupancy', 0)),
            float(data.get('co2_level', 400.0)),
        ]
    
    def _perform_prediction(self, sensor_data: Dict[str, Any]) -> Optional[PredictionResult]:
        """执行能耗预测"""
        if not self.energy_predictor:
            return None
        
        try:
            # 需要足够的历史数据
            if len(self._sensor_data_buffer) < self.energy_predictor.seq_length:
                return None
            
            # 转换为DataFrame
            df = pd.DataFrame(self._sensor_data_buffer)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # 确保必要的列存在
            for col in ['outdoor_temp', 'indoor_temp', 'hvac_power']:
                if col not in df.columns:
                    return None
            
            # 执行预测
            prediction = self.energy_predictor.predict(
                data=df,
                horizon=self.config.controller.prediction_horizon,
                target_column='hvac_power'
            )
            
            self.status.predictions_made += 1
            
            return prediction
        
        except Exception as e:
            logger.warning(f"Prediction failed: {e}")
            return None
    
    def _get_optimization_advice(self, sensor_data: Dict[str, Any]) -> Optional[str]:
        """从知识库获取优化建议"""
        if not self.knowledge_base:
            return None
        
        try:
            context = {
                'building_type': '办公楼',
                'current_temp': sensor_data.get('indoor_temp', 24.0),
                'target_temp': self.config.hvac.target_temperature,
                'occupancy': sensor_data.get('occupancy', 0),
                'time_of_day': datetime.now().strftime('%H:%M'),
                'energy_consumption': sensor_data.get('hvac_power', 0),
            }
            
            advice = self.knowledge_base.get_optimization_advice(context)
            return advice
        
        except Exception as e:
            logger.warning(f"Failed to get optimization advice: {e}")
            return None
    
    def _make_control_decision(
        self,
        sensor_data: Dict[str, Any],
        prediction: Optional[PredictionResult],
        advice: Optional[str]
    ) -> ControlDecision:
        """生成控制决策"""
        current_temp = sensor_data.get('indoor_temp', 24.0)
        target_temp = self.config.hvac.target_temperature
        occupancy = sensor_data.get('occupancy', 0)
        
        # 基础控制逻辑
        temp_diff = current_temp - target_temp
        
        if abs(temp_diff) <= self.config.hvac.comfort_tolerance:
            action = "maintain"
            new_target = target_temp
            reason = "Temperature within comfort zone"
        elif temp_diff > 0:
            action = "cool"
            new_target = max(target_temp - 1, self.config.hvac.min_setpoint)
            reason = f"Temperature {current_temp:.1f}°C above target"
        else:
            action = "heat"
            new_target = min(target_temp + 1, self.config.hvac.max_setpoint)
            reason = f"Temperature {current_temp:.1f}°C below target"
        
        # 考虑人员占用
        if occupancy == 0 and action == "maintain":
            # 无人时节能模式
            action = "eco_mode"
            new_target = target_temp + 2 if temp_diff > 0 else target_temp - 2
            new_target = max(self.config.hvac.min_setpoint, 
                           min(self.config.hvac.max_setpoint, new_target))
            reason += ", switching to eco mode (unoccupied)"
        
        decision = ControlDecision(
            timestamp=datetime.now(),
            action=action,
            target_temperature=new_target,
            hvac_power=self.config.hvac.hvac_power,
            reason=reason,
            confidence=0.8
        )
        
        self._control_history.append(decision)
        
        # 限制历史记录大小
        if len(self._control_history) > 1000:
            self._control_history = self._control_history[-1000:]
        
        return decision
    
    def _execute_control_action(self, decision: ControlDecision) -> None:
        """执行控制动作"""
        logger.debug(f"Executing control action: {decision.action}, "
                    f"target={decision.target_temperature}")
        
        if self.simulator:
            try:
                self.simulator.set_temperature_setpoint(decision.target_temperature)
            except Exception as e:
                logger.warning(f"Failed to set simulator setpoint: {e}")
        
        elif self.hvac_env:
            try:
                action = [decision.target_temperature]
                self.hvac_env.step(action)
            except Exception as e:
                logger.warning(f"Failed to execute HVAC action: {e}")
    
    def _record_data(
        self,
        sensor_data: Dict[str, Any],
        decision: ControlDecision,
        prediction: Optional[PredictionResult]
    ) -> None:
        """记录数据"""
        # 可以扩展为写入数据库或文件
        pass
    
    def _check_autosave(self) -> None:
        """检查并执行自动保存"""
        now = datetime.now()
        interval = timedelta(minutes=self.config.controller.autosave_interval)
        
        if now - self._last_autosave >= interval:
            self.save_state()
            self._last_autosave = now
    
    def pause(self) -> None:
        """暂停控制循环"""
        if self._running and not self._paused:
            self._paused = True
            self.state = SystemState.PAUSED
            logger.info("Controller paused")
    
    def resume(self) -> None:
        """恢复控制循环"""
        if self._running and self._paused:
            self._paused = False
            self.state = SystemState.RUNNING
            logger.info("Controller resumed")
    
    def shutdown(self, timeout: int = 30) -> None:
        """
        关闭系统
        
        Args:
            timeout: 等待关闭的超时时间（秒）
        """
        if not self._running and self.state == SystemState.STOPPED:
            return
        
        logger.info("Shutting down Building Energy Management System...")
        self.state = SystemState.SHUTTING_DOWN
        
        # 设置关闭事件
        self._shutdown_event.set()
        self._running = False
        
        # 保存状态
        try:
            self.save_state()
        except Exception as e:
            logger.warning(f"Failed to save state during shutdown: {e}")
        
        # 等待控制线程结束
        if self._control_thread and self._control_thread.is_alive():
            self._control_thread.join(timeout=timeout)
        
        # 关闭模块
        self._shutdown_modules()
        
        self.state = SystemState.STOPPED
        logger.info("System shutdown completed")
    
    def _shutdown_modules(self) -> None:
        """关闭各模块"""
        # 关闭模拟器
        if self.simulator:
            try:
                self.simulator.close()
            except Exception as e:
                logger.warning(f"Error closing simulator: {e}")
        
        # 关闭HVAC环境
        if self.hvac_env:
            try:
                self.hvac_env.close()
            except Exception as e:
                logger.warning(f"Error closing HVAC environment: {e}")
    
    def save_state(self) -> None:
        """保存系统状态"""
        state_dir = Path(self.config.system.output_dir) / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        
        state = {
            'timestamp': datetime.now().isoformat(),
            'status': self.status.to_dict(),
            'config': self.config.to_dict(),
            'control_history': [d.to_dict() for d in self._control_history[-100:]],
        }
        
        state_file = state_dir / f"state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"System state saved to {state_file}")
    
    def get_status(self) -> SystemStatus:
        """获取系统状态"""
        # 更新运行时间
        if self.status.start_time:
            self.status.uptime_seconds = (
                datetime.now() - self.status.start_time
            ).total_seconds()
        
        return self.status
    
    def get_module_status(self) -> Dict[str, Any]:
        """获取各模块状态"""
        return {
            'anomaly_detector': {
                'available': ANOMALY_AVAILABLE,
                'initialized': self.anomaly_detector is not None,
                'is_fitted': self.anomaly_detector.is_fitted if self.anomaly_detector else False,
            },
            'knowledge_base': {
                'available': KNOWLEDGE_AVAILABLE,
                'initialized': self.knowledge_base is not None,
                'indexed': self.knowledge_base._indexed if self.knowledge_base else False,
            },
            'energy_predictor': {
                'available': PREDICTOR_AVAILABLE,
                'initialized': self.energy_predictor is not None,
            },
            'simulator': {
                'available': SIMULATOR_AVAILABLE,
                'initialized': self.simulator is not None,
            },
            'hvac_env': {
                'available': HVAC_ENV_AVAILABLE,
                'initialized': self.hvac_env is not None,
            },
        }
    
    def query_knowledge_base(self, question: str) -> Optional[QueryResult]:
        """
        查询知识库
        
        Args:
            question: 问题文本
        
        Returns:
            查询结果
        """
        if not self.knowledge_base:
            logger.warning("Knowledge base not initialized")
            return None
        
        try:
            return self.knowledge_base.query(question)
        except Exception as e:
            logger.error(f"Knowledge base query failed: {e}")
            return None
    
    def get_alerts(self, severity: Optional[str] = None) -> List[Any]:
        """
        获取告警列表
        
        Args:
            severity: 按严重程度过滤
        
        Returns:
            告警列表
        """
        if self.anomaly_detector:
            return self.anomaly_detector.get_alerts(severity=severity)
        return []
    
    def register_cycle_callback(self, callback: Callable) -> None:
        """
        注册控制周期回调
        
        Args:
            callback: 回调函数，接收(sensor_data, decision)参数
        """
        self._cycle_callbacks.append(callback)
    
    def register_alert_callback(self, callback: Callable) -> None:
        """
        注册告警回调
        
        Args:
            callback: 回调函数，接收(alert)参数
        """
        self._alert_callbacks.append(callback)
    
    def _trigger_cycle_callbacks(self, sensor_data: Dict[str, Any], decision: ControlDecision) -> None:
        """触发周期回调"""
        for callback in self._cycle_callbacks:
            try:
                callback(sensor_data, decision)
            except Exception as e:
                logger.warning(f"Cycle callback failed: {e}")
    
    def _trigger_alert_callbacks(self, alert: Any) -> None:
        """触发告警回调"""
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.warning(f"Alert callback failed: {e}")


def create_controller(config_path: Optional[str] = None) -> BuildingController:
    """
    创建控制器实例的工厂函数
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        BuildingController实例
    """
    return BuildingController(config_path)


# 全局控制器实例
_controller_instance: Optional[BuildingController] = None


def get_controller(config_path: Optional[str] = None) -> BuildingController:
    """
    获取全局控制器实例（单例模式）
    
    Args:
        config_path: 配置文件路径，首次调用时有效
    
    Returns:
        BuildingController实例
    """
    global _controller_instance
    
    if _controller_instance is None:
        _controller_instance = create_controller(config_path)
    
    return _controller_instance


def reset_controller() -> None:
    """重置全局控制器实例"""
    global _controller_instance
    _controller_instance = None
