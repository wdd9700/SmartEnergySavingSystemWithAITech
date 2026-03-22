"""
偏差分析器

负责计算预测值与实际值之间的偏差度量指标，
包括MAE、RMSE、最大偏差等，并评估历史拟合度。
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np

try:
    from .predictor_monitor import PredictionActualPair
except ImportError:
    # 直接导入时回退到绝对导入
    from predictor_monitor import PredictionActualPair

logger = logging.getLogger(__name__)


@dataclass
class DeviationMetrics:
    """偏差度量"""
    # 温度相关指标
    temp_mae: float           # 温度平均绝对误差 (°C)
    temp_rmse: float          # 温度均方根误差 (°C)
    temp_max_dev: float       # 温度最大偏差 (°C)
    temp_mean_dev: float      # 温度平均偏差 (°C)
    temp_std: float           # 温度偏差标准差
    
    # 湿度相关指标
    humidity_mae: float       # 湿度平均绝对误差 (%)
    humidity_rmse: float      # 湿度均方根误差 (%)
    humidity_max_dev: float   # 湿度最大偏差 (%)
    
    # 功耗相关指标
    power_mae: float          # 功耗平均绝对误差 (kW)
    power_rmse: float         # 功耗均方根误差 (kW)
    power_max_dev: float      # 功耗最大偏差 (kW)
    power_relative_dev: float # 功耗相对偏差百分比 (%)
    
    # 综合指标
    relative_deviation: float # 综合相对偏差分数 (0-1)
    sample_count: int         # 样本数量
    time_window_hours: float  # 时间窗口 (小时)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'temperature': {
                'mae': round(self.temp_mae, 3),
                'rmse': round(self.temp_rmse, 3),
                'max_deviation': round(self.temp_max_dev, 3),
                'mean_deviation': round(self.temp_mean_dev, 3),
                'std': round(self.temp_std, 3)
            },
            'humidity': {
                'mae': round(self.humidity_mae, 3),
                'rmse': round(self.humidity_rmse, 3),
                'max_deviation': round(self.humidity_max_dev, 3)
            },
            'power': {
                'mae': round(self.power_mae, 3),
                'rmse': round(self.power_rmse, 3),
                'max_deviation': round(self.power_max_dev, 3),
                'relative_deviation': round(self.power_relative_dev, 3)
            },
            'overall': {
                'relative_deviation': round(self.relative_deviation, 3),
                'sample_count': self.sample_count,
                'time_window_hours': round(self.time_window_hours, 2)
            }
        }


class DeviationAnalyzer:
    """
    偏差分析器
    
    计算预测值与实际值之间的偏差度量指标，
    支持滑动窗口统计和历史拟合度评估。
    
    Attributes:
        window_size: 滑动窗口大小（样本数）
        temp_threshold: 温度偏差阈值 (°C)
        humidity_threshold: 湿度偏差阈值 (%)
        power_threshold: 功耗偏差阈值 (kW)
        historical_fit_threshold: 历史拟合度阈值
    
    Example:
        >>> analyzer = DeviationAnalyzer(window_size=168)
        >>> metrics = analyzer.calculate_metrics(history)
        >>> is_significant = analyzer.is_deviation_significant(metrics)
        >>> fit_score = analyzer.assess_historical_fit(history)
    """
    
    def __init__(
        self,
        window_size: int = 168,           # 默认7天(每小时采样)
        temp_threshold: float = 2.0,      # 温度偏差阈值 (°C)
        humidity_threshold: float = 10.0, # 湿度偏差阈值 (%)
        power_threshold: float = 1.0,     # 功耗偏差阈值 (kW)
        historical_fit_threshold: float = 0.7  # 历史拟合度阈值
    ):
        """
        初始化偏差分析器
        
        Args:
            window_size: 滑动窗口大小（样本数）
            temp_threshold: 温度偏差阈值
            humidity_threshold: 湿度偏差阈值
            power_threshold: 功耗偏差阈值
            historical_fit_threshold: 历史拟合度阈值
        """
        self.window_size = window_size
        self.temp_threshold = temp_threshold
        self.humidity_threshold = humidity_threshold
        self.power_threshold = power_threshold
        self.historical_fit_threshold = historical_fit_threshold
        
        # 历史拟合度缓存
        self._fitness_cache: Dict[str, Tuple[float, datetime]] = {}
        
        logger.info(f"DeviationAnalyzer initialized (window_size={window_size})")
    
    def calculate_metrics(
        self,
        history: List[PredictionActualPair],
        zone_id: Optional[str] = None
    ) -> DeviationMetrics:
        """
        计算偏差度量指标
        
        Args:
            history: 历史数据对列表
            zone_id: 指定区域，None表示所有区域
            
        Returns:
            DeviationMetrics对象
        """
        # 过滤指定区域的数据
        if zone_id:
            data = [h for h in history if h.zone_id == zone_id]
        else:
            data = history
        
        # 限制窗口大小
        if len(data) > self.window_size:
            data = data[-self.window_size:]
        
        if len(data) < 2:
            logger.warning(f"Insufficient data for metrics calculation: {len(data)} samples")
            return self._empty_metrics()
        
        # 提取数值
        pred_temps = np.array([d.predicted_temp for d in data])
        actual_temps = np.array([d.actual_temp for d in data])
        pred_humidities = np.array([d.predicted_humidity for d in data])
        actual_humidities = np.array([d.actual_humidity for d in data])
        pred_powers = np.array([d.predicted_power for d in data])
        actual_powers = np.array([d.actual_power for d in data])
        
        # 计算温度指标
        temp_errors = pred_temps - actual_temps
        temp_mae = np.mean(np.abs(temp_errors))
        temp_rmse = np.sqrt(np.mean(temp_errors ** 2))
        temp_max_dev = np.max(np.abs(temp_errors))
        temp_mean_dev = np.mean(temp_errors)
        temp_std = np.std(temp_errors)
        
        # 计算湿度指标
        humidity_errors = pred_humidities - actual_humidities
        humidity_mae = np.mean(np.abs(humidity_errors))
        humidity_rmse = np.sqrt(np.mean(humidity_errors ** 2))
        humidity_max_dev = np.max(np.abs(humidity_errors))
        
        # 计算功耗指标
        power_errors = pred_powers - actual_powers
        power_mae = np.mean(np.abs(power_errors))
        power_rmse = np.sqrt(np.mean(power_errors ** 2))
        power_max_dev = np.max(np.abs(power_errors))
        
        # 计算功耗相对偏差（避免除以0）
        mean_actual_power = np.mean(actual_powers)
        if mean_actual_power > 0.1:
            power_relative_dev = np.mean(np.abs(power_errors)) / mean_actual_power * 100
        else:
            power_relative_dev = 0.0
        
        # 计算综合相对偏差分数 (0-1)
        # 归一化各指标
        temp_score = min(temp_mae / self.temp_threshold, 1.0)
        humidity_score = min(humidity_mae / self.humidity_threshold, 1.0)
        power_score = min(power_mae / self.power_threshold, 1.0)
        
        # 加权平均（温度权重最高）
        relative_deviation = (
            0.5 * temp_score +
            0.2 * humidity_score +
            0.3 * power_score
        )
        
        # 计算时间窗口
        if len(data) >= 2:
            time_span = data[-1].timestamp - data[0].timestamp
            time_window_hours = time_span.total_seconds() / 3600
        else:
            time_window_hours = 0.0
        
        return DeviationMetrics(
            temp_mae=temp_mae,
            temp_rmse=temp_rmse,
            temp_max_dev=temp_max_dev,
            temp_mean_dev=temp_mean_dev,
            temp_std=temp_std,
            humidity_mae=humidity_mae,
            humidity_rmse=humidity_rmse,
            humidity_max_dev=humidity_max_dev,
            power_mae=power_mae,
            power_rmse=power_rmse,
            power_max_dev=power_max_dev,
            power_relative_dev=power_relative_dev,
            relative_deviation=relative_deviation,
            sample_count=len(data),
            time_window_hours=time_window_hours
        )
    
    def is_deviation_significant(
        self,
        metrics: DeviationMetrics,
        check_historical_fit: bool = True
    ) -> bool:
        """
        判断偏差是否显著（可能表示故障）
        
        判定条件:
        1. 当前偏差程度 > 阈值
        2. 历史拟合度良好（如果check_historical_fit为True）
        
        Args:
            metrics: 偏差度量指标
            check_historical_fit: 是否检查历史拟合度
            
        Returns:
            True表示偏差显著，可能存在故障
        """
        # 检查温度偏差
        temp_significant = metrics.temp_mae > self.temp_threshold
        
        # 检查湿度偏差
        humidity_significant = metrics.humidity_mae > self.humidity_threshold
        
        # 检查功耗偏差
        power_significant = metrics.power_mae > self.power_threshold
        
        # 综合判断
        is_significant = temp_significant or humidity_significant or power_significant
        
        if is_significant:
            logger.info(f"Significant deviation detected: "
                       f"temp_mae={metrics.temp_mae:.2f}°C, "
                       f"humidity_mae={metrics.humidity_mae:.2f}%, "
                       f"power_mae={metrics.power_mae:.2f}kW")
        
        return is_significant
    
    def assess_historical_fit(
        self,
        history: List[PredictionActualPair],
        zone_id: Optional[str] = None,
        reference_window_hours: int = 168  # 默认7天
    ) -> float:
        """
        评估历史拟合度
        
        计算过去一段时间内的预测准确度，用于判断模型是否可靠。
        返回拟合度分数 (0-1)，越接近1表示拟合越好。
        
        Args:
            history: 历史数据对列表
            zone_id: 指定区域
            reference_window_hours: 参考时间窗口（小时）
            
        Returns:
            拟合度分数 (0-1)
        """
        cache_key = zone_id or 'all'
        
        # 检查缓存
        if cache_key in self._fitness_cache:
            cached_score, cached_time = self._fitness_cache[cache_key]
            if datetime.now() - cached_time < timedelta(minutes=5):
                return cached_score
        
        # 过滤数据
        if zone_id:
            data = [h for h in history if h.zone_id == zone_id]
        else:
            data = history
        
        # 获取参考窗口的数据（排除最近的数据）
        cutoff_time = datetime.now() - timedelta(hours=reference_window_hours)
        reference_data = [h for h in data if h.timestamp < cutoff_time]
        
        if len(reference_data) < 24:  # 至少需要1天的数据
            logger.warning(f"Insufficient historical data for fit assessment: {len(reference_data)} samples")
            # 如果没有足够历史数据，假设拟合度良好
            return 0.8
        
        # 计算历史MAE
        pred_temps = np.array([d.predicted_temp for d in reference_data])
        actual_temps = np.array([d.actual_temp for d in reference_data])
        temp_errors = np.abs(pred_temps - actual_temps)
        historical_mae = np.mean(temp_errors)
        
        # 计算拟合度分数
        # MAE < 0.5°C -> 拟合度 1.0
        # MAE > 3.0°C -> 拟合度 0.0
        if historical_mae <= 0.5:
            fit_score = 1.0
        elif historical_mae >= 3.0:
            fit_score = 0.0
        else:
            fit_score = (3.0 - historical_mae) / 2.5
        
        # 缓存结果
        self._fitness_cache[cache_key] = (fit_score, datetime.now())
        
        logger.debug(f"Historical fit score for {cache_key}: {fit_score:.3f} (MAE={historical_mae:.2f}°C)")
        
        return fit_score
    
    def is_model_reliable(
        self,
        history: List[PredictionActualPair],
        zone_id: Optional[str] = None
    ) -> bool:
        """
        判断模型是否可靠（历史拟合度是否良好）
        
        Args:
            history: 历史数据对列表
            zone_id: 指定区域
            
        Returns:
            True表示模型可靠
        """
        fit_score = self.assess_historical_fit(history, zone_id)
        return fit_score >= self.historical_fit_threshold
    
    def get_deviation_trend(
        self,
        history: List[PredictionActualPair],
        window_size: int = 24,
        zone_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        获取偏差趋势分析
        
        Args:
            history: 历史数据对列表
            window_size: 趋势窗口大小（样本数）
            zone_id: 指定区域
            
        Returns:
            趋势分析结果字典
        """
        # 过滤数据
        if zone_id:
            data = [h for h in history if h.zone_id == zone_id]
        else:
            data = history
        
        if len(data) < window_size * 2:
            return {
                'trend': 'insufficient_data',
                'temp_trend': 0.0,
                'description': '数据不足，无法分析趋势'
            }
        
        # 分窗口计算
        recent_data = data[-window_size:]
        previous_data = data[-window_size*2:-window_size]
        
        recent_metrics = self.calculate_metrics(recent_data)
        previous_metrics = self.calculate_metrics(previous_data)
        
        # 计算趋势
        temp_trend = recent_metrics.temp_mae - previous_metrics.temp_mae
        
        if abs(temp_trend) < 0.2:
            trend = 'stable'
            description = '偏差稳定'
        elif temp_trend > 0:
            trend = 'increasing'
            description = f'偏差呈上升趋势 (+{temp_trend:.2f}°C)'
        else:
            trend = 'decreasing'
            description = f'偏差呈下降趋势 ({temp_trend:.2f}°C)'
        
        return {
            'trend': trend,
            'temp_trend': temp_trend,
            'recent_mae': recent_metrics.temp_mae,
            'previous_mae': previous_metrics.temp_mae,
            'description': description
        }
    
    def _empty_metrics(self) -> DeviationMetrics:
        """返回空的度量指标"""
        return DeviationMetrics(
            temp_mae=0.0,
            temp_rmse=0.0,
            temp_max_dev=0.0,
            temp_mean_dev=0.0,
            temp_std=0.0,
            humidity_mae=0.0,
            humidity_rmse=0.0,
            humidity_max_dev=0.0,
            power_mae=0.0,
            power_rmse=0.0,
            power_max_dev=0.0,
            power_relative_dev=0.0,
            relative_deviation=0.0,
            sample_count=0,
            time_window_hours=0.0
        )
    
    def update_thresholds(
        self,
        temp_threshold: Optional[float] = None,
        humidity_threshold: Optional[float] = None,
        power_threshold: Optional[float] = None,
        historical_fit_threshold: Optional[float] = None
    ) -> None:
        """
        更新阈值参数
        
        Args:
            temp_threshold: 温度偏差阈值
            humidity_threshold: 湿度偏差阈值
            power_threshold: 功耗偏差阈值
            historical_fit_threshold: 历史拟合度阈值
        """
        if temp_threshold is not None:
            self.temp_threshold = temp_threshold
        if humidity_threshold is not None:
            self.humidity_threshold = humidity_threshold
        if power_threshold is not None:
            self.power_threshold = power_threshold
        if historical_fit_threshold is not None:
            self.historical_fit_threshold = historical_fit_threshold
        
        logger.info(f"Thresholds updated: temp={self.temp_threshold}, "
                   f"humidity={self.humidity_threshold}, power={self.power_threshold}")
