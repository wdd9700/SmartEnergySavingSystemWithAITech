"""
建筑设备异常检测模块

基于PyOD库实现多种异常检测算法，用于检测建筑设备运行异常。
支持Isolation Forest、AutoEncoder等算法，提供与HVAC系统的集成接口。
"""

import os
import logging
import pickle
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from scipy.special import expit

# PyOD异常检测库
try:
    from pyod.models.iforest import IForest
    from pyod.models.auto_encoder import AutoEncoder
    from pyod.models.lof import LOF
    from pyod.models.hbos import HBOS
    PYOD_AVAILABLE = True
except ImportError:
    PYOD_AVAILABLE = False
    logging.warning("PyOD not installed. Anomaly detection functionality will be limited.")

# sklearn用于数据预处理
try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not installed. Data preprocessing will be limited.")

logger = logging.getLogger(__name__)


@dataclass
class AnomalyAlert:
    """异常告警数据结构"""
    timestamp: datetime
    anomaly_score: float
    anomaly_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    affected_metrics: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'anomaly_score': self.anomaly_score,
            'anomaly_type': self.anomaly_type,
            'severity': self.severity,
            'description': self.description,
            'affected_metrics': self.affected_metrics
        }


class AnomalyDetector:
    """
    建筑设备异常检测器
    
    基于PyOD实现多种异常检测算法，支持历史数据训练和实时检测。
    
    Attributes:
        algorithm: 检测算法类型 ('iforest', 'autoencoder', 'lof', 'hbos')
        contamination: 异常数据比例估计 (0.0-0.5)
        model: 训练好的检测模型
        scaler: 数据标准化器
        feature_names: 特征名称列表
        alert_threshold: 告警阈值
        alert_history: 告警历史记录
    
    Example:
        >>> detector = AnomalyDetector(algorithm='iforest', contamination=0.1)
        >>> detector.fit(training_data)
        >>> predictions = detector.predict(real_time_data)
        >>> alerts = detector.get_alerts()
    """
    
    SUPPORTED_ALGORITHMS = ['iforest', 'autoencoder', 'lof', 'hbos']
    
    def __init__(
        self,
        algorithm: str = 'iforest',
        contamination: float = 0.1,
        alert_threshold: float = 0.8,
        feature_names: Optional[List[str]] = None,
        model_params: Optional[Dict[str, Any]] = None
    ):
        """
        初始化异常检测器
        
        Args:
            algorithm: 检测算法类型，支持 'iforest', 'autoencoder', 'lof', 'hbos'
            contamination: 异常数据比例估计，范围 (0.0, 0.5]
            alert_threshold: 告警阈值，异常分数超过此值触发告警 (0.0-1.0)
            feature_names: 特征名称列表，用于告警描述
            model_params: 模型额外参数
        
        Raises:
            ValueError: 当algorithm不支持或contamination超出范围
            ImportError: 当PyOD未安装
        """
        if not PYOD_AVAILABLE:
            raise ImportError("PyOD is required. Install with: pip install pyod")
        
        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm: {algorithm}. "
                f"Supported: {self.SUPPORTED_ALGORITHMS}"
            )
        
        if not 0.0 < contamination <= 0.5:
            raise ValueError("contamination must be in (0.0, 0.5]")
        
        self.algorithm = algorithm
        self.contamination = contamination
        self.alert_threshold = alert_threshold
        self.feature_names = feature_names or []
        self.model_params = model_params or {}
        
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.is_fitted = False
        self.alert_history: List[AnomalyAlert] = []
        self.training_stats: Dict[str, Any] = {}
        
        logger.info(f"AnomalyDetector initialized with {algorithm} algorithm")
    
    def _create_model(self) -> Any:
        """
        创建检测模型实例
        
        Returns:
            PyOD模型实例
        """
        if self.algorithm == 'iforest':
            params = {
                'n_estimators': 100,
                'contamination': self.contamination,
                'random_state': 42,
                **self.model_params
            }
            return IForest(**params)
        
        elif self.algorithm == 'autoencoder':
            params = {
                'contamination': self.contamination,
                'hidden_neurons': [64, 32, 32, 64],
                'epochs': 100,
                'batch_size': 32,
                'verbose': 0,
                **self.model_params
            }
            return AutoEncoder(**params)
        
        elif self.algorithm == 'lof':
            params = {
                'n_neighbors': 20,
                'contamination': self.contamination,
                **self.model_params
            }
            return LOF(**params)
        
        elif self.algorithm == 'hbos':
            params = {
                'n_bins': 10,
                'contamination': self.contamination,
                **self.model_params
            }
            return HBOS(**params)
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        训练异常检测模型
        
        Args:
            X: 训练数据，形状为 (n_samples, n_features)
            y: 标签数据（可选，用于有监督场景）
        
        Raises:
            ValueError: 当输入数据格式不正确
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")
        
        if len(X) < 10:
            raise ValueError(f"Training data too small: {len(X)} samples, need at least 10")
        
        logger.info(f"Training {self.algorithm} model on {X.shape[0]} samples...")
        
        # 数据标准化
        if self.scaler is not None:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        
        # 创建并训练模型
        self.model = self._create_model()
        self.model.fit(X_scaled)
        
        # 记录训练统计
        self.training_stats = {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'algorithm': self.algorithm,
            'contamination': self.contamination,
            'training_time': datetime.now().isoformat()
        }
        
        self.is_fitted = True
        logger.info(f"Model training completed. Features: {X.shape[1]}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测异常标签
        
        Args:
            X: 输入数据，形状为 (n_samples, n_features)
        
        Returns:
            异常标签数组，1表示异常，0表示正常
        
        Raises:
            RuntimeError: 当模型未训练时
        """
        self._check_fitted()
        
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        predictions = self.model.predict(X_scaled)
        
        # 生成告警
        self._generate_alerts(X, predictions)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测异常概率
        
        Args:
            X: 输入数据，形状为 (n_samples, n_features)
        
        Returns:
            异常概率数组，形状为 (n_samples, 2)，
            第0列为正常概率，第1列为异常概率
        
        Raises:
            RuntimeError: 当模型未训练时
        """
        self._check_fitted()
        
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # PyOD的predict_proba返回异常概率
        try:
            proba = self.model.predict_proba(X_scaled)
        except AttributeError:
            # 某些模型可能不支持predict_proba，使用decision_function
            scores = self.model.decision_function(X_scaled)
            # 将分数转换为概率
            proba = self._scores_to_proba(scores)
        
        return proba
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        计算异常分数
        
        Args:
            X: 输入数据
        
        Returns:
            异常分数数组，分数越高表示越可能是异常
        """
        self._check_fitted()
        
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.model.decision_function(X_scaled)
    
    def _scores_to_proba(self, scores: np.ndarray) -> np.ndarray:
        """
        将异常分数转换为概率
        
        Args:
            scores: 异常分数
        
        Returns:
            概率数组
        """
        # 使用sigmoid转换 (expit已在模块顶部导入)
        # 标准化分数
        scores_normalized = (scores - scores.mean()) / (scores.std() + 1e-10)
        anomaly_proba = expit(scores_normalized)
        
        return np.column_stack([1 - anomaly_proba, anomaly_proba])
    
    def _generate_alerts(
        self,
        X: np.ndarray,
        predictions: np.ndarray
    ) -> None:
        """
        生成异常告警
        
        Args:
            X: 输入数据
            predictions: 预测标签
        """
        anomaly_indices = np.where(predictions == 1)[0]
        
        if len(anomaly_indices) == 0:
            return
        
        # 计算异常分数
        scores = self.decision_function(X)
        
        for idx in anomaly_indices:
            if idx >= len(scores):
                continue
            score = scores[idx]
            
            # 归一化分数到0-1
            score_normalized = min(max(score / 10.0, 0.0), 1.0)
            
            # 确定严重程度
            if score_normalized >= 0.9:
                severity = 'critical'
            elif score_normalized >= 0.7:
                severity = 'high'
            elif score_normalized >= 0.5:
                severity = 'medium'
            else:
                severity = 'low'
            
            # 只记录超过阈值的告警
            if score_normalized >= self.alert_threshold:
                # 识别受影响的指标
                affected_metrics = self._identify_affected_metrics(X[idx])
                
                alert = AnomalyAlert(
                    timestamp=datetime.now(),
                    anomaly_score=float(score_normalized),
                    anomaly_type=self._classify_anomaly_type(X[idx]),
                    severity=severity,
                    description=self._generate_alert_description(
                        X[idx], score_normalized, affected_metrics
                    ),
                    affected_metrics=affected_metrics
                )
                
                self.alert_history.append(alert)
                logger.warning(f"Anomaly detected: {alert.description}")
    
    def _identify_affected_metrics(self, x: np.ndarray) -> List[str]:
        """
        识别受影响的指标
        
        Args:
            x: 单条数据
        
        Returns:
            受影响的指标名称列表
        """
        if not self.feature_names or len(self.feature_names) != len(x):
            return ['unknown']
        
        # 找出偏离正常范围最大的特征
        # 这里简化处理，假设训练数据已标准化
        affected = []
        for i, (name, value) in enumerate(zip(self.feature_names, x)):
            if abs(value) > 2.0:  # 超过2个标准差
                affected.append(name)
        
        return affected if affected else ['multiple_metrics']
    
    def _classify_anomaly_type(self, x: np.ndarray) -> str:
        """
        分类异常类型
        
        Args:
            x: 单条数据
        
        Returns:
            异常类型描述
        """
        # 基于特征值判断异常类型
        if not self.feature_names:
            return 'general_anomaly'
        
        # HVAC相关异常检测
        if 'hvac_power' in self.feature_names:
            idx = self.feature_names.index('hvac_power')
            if x[idx] > 3.0:  # 高功耗
                return 'hvac_high_power'
            elif x[idx] < -2.0:  # 异常低功耗
                return 'hvac_low_power'
        
        if 'indoor_temp' in self.feature_names:
            idx = self.feature_names.index('indoor_temp')
            if abs(x[idx]) > 2.0:
                return 'temperature_anomaly'
        
        return 'general_anomaly'
    
    def _generate_alert_description(
        self,
        x: np.ndarray,
        score: float,
        affected_metrics: List[str]
    ) -> str:
        """
        生成告警描述
        
        Args:
            x: 单条数据
            score: 异常分数
            affected_metrics: 受影响的指标
        
        Returns:
            告警描述文本
        """
        anomaly_type = self._classify_anomaly_type(x)
        
        descriptions = {
            'hvac_high_power': f"HVAC系统功耗异常升高，异常分数: {score:.2f}",
            'hvac_low_power': f"HVAC系统功耗异常降低，可能存在故障，异常分数: {score:.2f}",
            'temperature_anomaly': f"室内温度异常，异常分数: {score:.2f}",
            'general_anomaly': f"检测到设备运行异常，异常分数: {score:.2f}"
        }
        
        base_desc = descriptions.get(anomaly_type, descriptions['general_anomaly'])
        
        if affected_metrics:
            base_desc += f"，受影响指标: {', '.join(affected_metrics)}"
        
        return base_desc
    
    def get_alerts(
        self,
        severity: Optional[str] = None,
        since: Optional[datetime] = None
    ) -> List[AnomalyAlert]:
        """
        获取告警历史
        
        Args:
            severity: 按严重程度过滤 ('low', 'medium', 'high', 'critical')
            since: 只返回此时间之后的告警
        
        Returns:
            告警列表
        """
        alerts = self.alert_history
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if since:
            alerts = [a for a in alerts if a.timestamp >= since]
        
        return alerts
    
    def clear_alerts(self) -> None:
        """清空告警历史"""
        self.alert_history.clear()
        logger.info("Alert history cleared")
    
    def save(self, path: str) -> None:
        """
        保存模型到文件
        
        Args:
            path: 保存路径
        
        Raises:
            RuntimeError: 当模型未训练时
        """
        self._check_fitted()
        
        # 创建目录
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        # 保存模型和配置
        save_data = {
            'model': self.model,
            'scaler': self.scaler,
            'algorithm': self.algorithm,
            'contamination': self.contamination,
            'alert_threshold': self.alert_threshold,
            'feature_names': self.feature_names,
            'training_stats': self.training_stats,
            'model_params': self.model_params
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """
        从文件加载模型
        
        Args:
            path: 模型文件路径
        
        Raises:
            FileNotFoundError: 当文件不存在时
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.model = save_data['model']
        self.scaler = save_data.get('scaler')
        self.algorithm = save_data['algorithm']
        self.contamination = save_data['contamination']
        self.alert_threshold = save_data.get('alert_threshold', 0.8)
        self.feature_names = save_data.get('feature_names', [])
        self.training_stats = save_data.get('training_stats', {})
        self.model_params = save_data.get('model_params', {})
        self.is_fitted = True
        
        logger.info(f"Model loaded from {path}")
    
    def _check_fitted(self) -> None:
        """
        检查模型是否已训练
        
        Raises:
            RuntimeError: 当模型未训练时
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        info = {
            'algorithm': self.algorithm,
            'contamination': self.contamination,
            'alert_threshold': self.alert_threshold,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'supported_algorithms': self.SUPPORTED_ALGORITHMS,
            'alert_count': len(self.alert_history)
        }
        
        if self.training_stats:
            info.update(self.training_stats)
        
        return info


class HVACAnomalyDetector(AnomalyDetector):
    """
    HVAC系统专用异常检测器
    
    针对HVAC系统特点优化的异常检测器，预定义了HVAC相关特征。
    
    Attributes:
        temp_threshold: 温度异常阈值
        power_threshold: 功耗异常阈值
    
    Example:
        >>> detector = HVACAnomalyDetector(algorithm='iforest')
        >>> detector.fit(hvac_data)
        >>> alerts = detector.monitor(hvac_realtime_data)
    """
    
    HVAC_FEATURES = [
        'outdoor_temp',
        'indoor_temp',
        'indoor_humidity',
        'hvac_power',
        'setpoint_temp',
        'occupancy',
        'hour',
        'solar_radiation'
    ]
    
    def __init__(
        self,
        algorithm: str = 'iforest',
        contamination: float = 0.1,
        alert_threshold: float = 0.8,
        temp_threshold: float = 2.0,
        power_threshold: float = 3.0,
        model_params: Optional[Dict[str, Any]] = None
    ):
        """
        初始化HVAC异常检测器
        
        Args:
            algorithm: 检测算法
            contamination: 异常比例
            alert_threshold: 告警阈值
            temp_threshold: 温度偏差阈值 (°C)
            power_threshold: 功耗异常阈值 (kW)
            model_params: 模型额外参数
        """
        super().__init__(
            algorithm=algorithm,
            contamination=contamination,
            alert_threshold=alert_threshold,
            feature_names=self.HVAC_FEATURES,
            model_params=model_params
        )
        
        self.temp_threshold = temp_threshold
        self.power_threshold = power_threshold
    
    def monitor(self, data: np.ndarray) -> Dict[str, Any]:
        """
        监控HVAC系统状态
        
        Args:
            data: 实时数据
        
        Returns:
            监控结果字典
        """
        predictions = self.predict(data)
        proba = self.predict_proba(data)
        scores = self.decision_function(data)
        
        # 获取最新告警
        recent_alerts = self.get_alerts(since=datetime.now().replace(minute=0, second=0))
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'is_anomaly': bool(predictions[-1]) if len(predictions) > 0 else False,
            'anomaly_probability': float(proba[-1, 1]) if len(proba) > 0 else 0.0,
            'anomaly_score': float(scores[-1]) if len(scores) > 0 else 0.0,
            'active_alerts': [a.to_dict() for a in recent_alerts[-5:]],
            'system_status': self._assess_system_status(data[-1] if len(data) > 0 else None)
        }
        
        return result
    
    def _assess_system_status(self, data_point: Optional[np.ndarray]) -> str:
        """
        评估HVAC系统状态
        
        Args:
            data_point: 单条数据
        
        Returns:
            状态描述
        """
        if data_point is None:
            return 'unknown'
        
        # 基于特征评估状态
        issues = []
        
        if len(data_point) > 3:
            hvac_power = data_point[3]
            if hvac_power > self.power_threshold:
                issues.append('high_power')
        
        if len(data_point) > 1:
            indoor_temp = data_point[1]
            if abs(indoor_temp) > self.temp_threshold:
                issues.append('temp_deviation')
        
        if not issues:
            return 'normal'
        elif len(issues) == 1:
            return f'warning: {issues[0]}'
        else:
            return f'critical: {", ".join(issues)}'


def create_detector_from_config(config: Dict[str, Any]) -> AnomalyDetector:
    """
    从配置创建检测器
    
    Args:
        config: 配置字典
    
    Returns:
        配置的检测器实例
    
    Example:
        >>> config = {
        ...     'algorithm': 'iforest',
        ...     'contamination': 0.1,
        ...     'alert_threshold': 0.8,
        ...     'feature_names': ['temp', 'power']
        ... }
        >>> detector = create_detector_from_config(config)
    """
    detector_type = config.get('type', 'general')
    
    if detector_type == 'hvac':
        return HVACAnomalyDetector(**{k: v for k, v in config.items() if k != 'type'})
    else:
        return AnomalyDetector(**{k: v for k, v in config.items() if k != 'type'})
