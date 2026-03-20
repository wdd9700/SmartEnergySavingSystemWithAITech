"""
能耗基线模型

提供多种基线模型用于能耗对比分析：
- 历史平均基线
- 回归基线
- 聚类基线
- 时间序列基线
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


class BaselineType(Enum):
    """基线模型类型"""
    HISTORICAL_AVERAGE = "historical_average"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    SEASONAL_NAIVE = "seasonal_naive"
    MOVING_AVERAGE = "moving_average"


@dataclass
class BaselineResult:
    """基线预测结果"""
    baseline_values: np.ndarray      # 基线预测值
    actual_values: np.ndarray        # 实际值
    savings: np.ndarray              # 节能量
    savings_percent: float           # 节能百分比
    mape: float                      # MAPE
    rmse: float                      # RMSE
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'baseline_values': self.baseline_values.tolist(),
            'actual_values': self.actual_values.tolist(),
            'savings': self.savings.tolist(),
            'savings_percent': self.savings_percent,
            'mape': self.mape,
            'rmse': self.rmse
        }


class HistoricalAverageBaseline:
    """
    历史平均基线模型
    
    基于历史同期数据计算基线。
    """
    
    def __init__(
        self,
        aggregation: str = "mean",
        lookback_days: int = 30,
        exclude_outliers: bool = True,
        outlier_threshold: float = 3.0
    ):
        """
        初始化历史平均基线模型
        
        Args:
            aggregation: 聚合方式 ("mean", "median")
            lookback_days: 回溯天数
            exclude_outliers: 是否排除异常值
            outlier_threshold: 异常值阈值（标准差倍数）
        """
        self.aggregation = aggregation
        self.lookback_days = lookback_days
        self.exclude_outliers = exclude_outliers
        self.outlier_threshold = outlier_threshold
        self.baseline_stats: Dict[int, float] = {}
    
    def fit(self, data: pd.DataFrame, target_column: str = "hvac_power") -> None:
        """
        训练基线模型
        
        Args:
            data: 历史数据DataFrame
            target_column: 目标列名
        """
        df = data.copy()
        
        # 确保时间索引
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        
        # 按小时和星期几分组计算基线
        for hour in range(24):
            for dow in range(7):
                mask = (df['hour'] == hour) & (df['day_of_week'] == dow)
                values = df.loc[mask, target_column].values
                
                if len(values) > 0:
                    # 排除异常值
                    if self.exclude_outliers and len(values) > 3:
                        mean = np.mean(values)
                        std = np.std(values)
                        values = values[
                            np.abs(values - mean) < self.outlier_threshold * std
                        ]
                    
                    # 计算基线
                    if self.aggregation == "mean":
                        baseline = np.mean(values)
                    elif self.aggregation == "median":
                        baseline = np.median(values)
                    else:
                        baseline = np.mean(values)
                    
                    key = hour * 7 + dow
                    self.baseline_stats[key] = baseline
        
        logger.info(f"HistoricalAverageBaseline fitted with {len(self.baseline_stats)} time slots")
    
    def predict(
        self,
        timestamps: List[datetime],
        target_column: str = "hvac_power"
    ) -> np.ndarray:
        """
        预测基线值
        
        Args:
            timestamps: 时间戳列表
            target_column: 目标列名
        
        Returns:
            基线预测值
        """
        predictions = []
        
        for ts in timestamps:
            key = ts.hour * 7 + ts.dayofweek
            baseline = self.baseline_stats.get(key, np.mean(list(self.baseline_stats.values())))
            predictions.append(baseline)
        
        return np.array(predictions)
    
    def save(self, path: str) -> None:
        """保存模型"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        save_data = {
            'baseline_stats': self.baseline_stats,
            'aggregation': self.aggregation,
            'lookback_days': self.lookback_days,
            'exclude_outliers': self.exclude_outliers,
            'outlier_threshold': self.outlier_threshold
        }
        
        with open(path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        logger.info(f"HistoricalAverageBaseline saved to {path}")
    
    def load(self, path: str) -> None:
        """加载模型"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.baseline_stats = {int(k): v for k, v in data['baseline_stats'].items()}
        self.aggregation = data['aggregation']
        self.lookback_days = data['lookback_days']
        self.exclude_outliers = data['exclude_outliers']
        self.outlier_threshold = data['outlier_threshold']
        
        logger.info(f"HistoricalAverageBaseline loaded from {path}")


class RegressionBaseline:
    """
    回归基线模型
    
    使用线性回归或岭回归建立能耗基线。
    """
    
    def __init__(
        self,
        model_type: str = "ridge",
        alpha: float = 1.0,
        feature_columns: Optional[List[str]] = None
    ):
        """
        初始化回归基线模型
        
        Args:
            model_type: 模型类型 ("linear", "ridge")
            alpha: 正则化强度（Ridge）
            feature_columns: 特征列名列表
        """
        self.model_type = model_type
        self.alpha = alpha
        self.feature_columns = feature_columns
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """准备特征数据"""
        df = data.copy()
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        # 时间特征
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        # 周期性编码
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def fit(self, data: pd.DataFrame, target_column: str = "hvac_power") -> None:
        """
        训练回归模型
        
        Args:
            data: 历史数据DataFrame
            target_column: 目标列名
        """
        df = self._prepare_features(data)
        
        # 确定特征列
        if self.feature_columns is None:
            self.feature_columns = [
                'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
                'is_weekend', 'month'
            ]
            # 添加天气相关特征
            for col in ['outdoor_temp', 'indoor_temp', 'indoor_humidity', 'solar_radiation']:
                if col in df.columns:
                    self.feature_columns.append(col)
        
        X = df[self.feature_columns].values
        y = df[target_column].values
        
        # 缩放特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练模型
        if self.model_type == "linear":
            self.model = LinearRegression()
        elif self.model_type == "ridge":
            self.model = Ridge(alpha=self.alpha)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        logger.info(f"RegressionBaseline fitted with {len(self.feature_columns)} features")
    
    def predict(
        self,
        data: pd.DataFrame,
        target_column: str = "hvac_power"
    ) -> np.ndarray:
        """
        预测基线值
        
        Args:
            data: 输入数据DataFrame
            target_column: 目标列名
        
        Returns:
            基线预测值
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        df = self._prepare_features(data)
        X = df[self.feature_columns].values
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        importance = dict(zip(self.feature_columns, self.model.coef_))
        return dict(sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True))
    
    def save(self, path: str) -> None:
        """保存模型"""
        import pickle
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        save_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'alpha': self.alpha
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"RegressionBaseline saved to {path}")
    
    def load(self, path: str) -> None:
        """加载模型"""
        import pickle
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        self.model_type = data['model_type']
        self.alpha = data['alpha']
        self.is_fitted = True
        
        logger.info(f"RegressionBaseline loaded from {path}")


class ClusteringBaseline:
    """
    聚类基线模型
    
    使用K-means聚类识别典型能耗模式。
    """
    
    def __init__(
        self,
        n_clusters: int = 5,
        feature_columns: Optional[List[str]] = None
    ):
        """
        初始化聚类基线模型
        
        Args:
            n_clusters: 聚类数量
            feature_columns: 特征列名列表
        """
        self.n_clusters = n_clusters
        self.feature_columns = feature_columns
        self.kmeans = None
        self.scaler = StandardScaler()
        self.cluster_profiles: Dict[int, Dict[str, float]] = {}
        self.is_fitted = False
    
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """准备特征数据"""
        df = data.copy()
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        # 使用小时级特征
        if self.feature_columns is None:
            self.feature_columns = ['hour', 'day_of_week', 'outdoor_temp']
            self.feature_columns = [c for c in self.feature_columns if c in df.columns]
        
        return df[self.feature_columns].values
    
    def fit(self, data: pd.DataFrame, target_column: str = "hvac_power") -> None:
        """
        训练聚类模型
        
        Args:
            data: 历史数据DataFrame
            target_column: 目标列名
        """
        X = self._prepare_features(data)
        X_scaled = self.scaler.fit_transform(X)
        
        # K-means聚类
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = self.kmeans.fit_predict(X_scaled)
        
        # 计算每个聚类的能耗特征
        df = data.copy()
        df['cluster'] = labels
        
        for cluster_id in range(self.n_clusters):
            cluster_data = df[df['cluster'] == cluster_id][target_column]
            self.cluster_profiles[cluster_id] = {
                'mean': float(cluster_data.mean()),
                'std': float(cluster_data.std()),
                'median': float(cluster_data.median()),
                'count': int(len(cluster_data))
            }
        
        self.is_fitted = True
        logger.info(f"ClusteringBaseline fitted with {self.n_clusters} clusters")
    
    def predict(
        self,
        data: pd.DataFrame,
        target_column: str = "hvac_power"
    ) -> np.ndarray:
        """
        预测基线值
        
        Args:
            data: 输入数据DataFrame
            target_column: 目标列名
        
        Returns:
            基线预测值
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        X = self._prepare_features(data)
        X_scaled = self.scaler.transform(X)
        
        labels = self.kmeans.predict(X_scaled)
        predictions = [self.cluster_profiles[l]['mean'] for l in labels]
        
        return np.array(predictions)
    
    def get_cluster_info(self) -> Dict[int, Dict[str, float]]:
        """获取聚类信息"""
        return self.cluster_profiles.copy()
    
    def save(self, path: str) -> None:
        """保存模型"""
        import pickle
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        save_data = {
            'kmeans': self.kmeans,
            'scaler': self.scaler,
            'cluster_profiles': self.cluster_profiles,
            'feature_columns': self.feature_columns,
            'n_clusters': self.n_clusters
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"ClusteringBaseline saved to {path}")
    
    def load(self, path: str) -> None:
        """加载模型"""
        import pickle
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.kmeans = data['kmeans']
        self.scaler = data['scaler']
        self.cluster_profiles = data['cluster_profiles']
        self.feature_columns = data['feature_columns']
        self.n_clusters = data['n_clusters']
        self.is_fitted = True
        
        logger.info(f"ClusteringBaseline loaded from {path}")


class BaselineModel:
    """
    能耗基线模型统一接口
    
    提供多种基线模型的统一接口，支持模型对比和节能分析。
    """
    
    def __init__(
        self,
        baseline_type: BaselineType = BaselineType.HISTORICAL_AVERAGE,
        **kwargs
    ):
        """
        初始化基线模型
        
        Args:
            baseline_type: 基线模型类型
            **kwargs: 传递给具体模型的参数
        """
        self.baseline_type = baseline_type
        self.model: Any = None
        self._create_model(**kwargs)
    
    def _create_model(self, **kwargs) -> None:
        """创建具体基线模型"""
        if self.baseline_type == BaselineType.HISTORICAL_AVERAGE:
            self.model = HistoricalAverageBaseline(**kwargs)
        elif self.baseline_type == BaselineType.REGRESSION:
            self.model = RegressionBaseline(**kwargs)
        elif self.baseline_type == BaselineType.CLUSTERING:
            self.model = ClusteringBaseline(**kwargs)
        else:
            raise ValueError(f"Unsupported baseline type: {self.baseline_type}")
    
    def fit(self, data: pd.DataFrame, target_column: str = "hvac_power") -> None:
        """训练基线模型"""
        self.model.fit(data, target_column)
    
    def predict(
        self,
        data: pd.DataFrame,
        target_column: str = "hvac_power"
    ) -> np.ndarray:
        """预测基线值"""
        if self.baseline_type == BaselineType.HISTORICAL_AVERAGE:
            if 'timestamp' in data.columns:
                timestamps = pd.to_datetime(data['timestamp'])
            else:
                timestamps = data.index
            return self.model.predict(timestamps.tolist(), target_column)
        else:
            return self.model.predict(data, target_column)
    
    def evaluate(
        self,
        data: pd.DataFrame,
        target_column: str = "hvac_power"
    ) -> BaselineResult:
        """
        评估基线模型并计算节能效果
        
        Args:
            data: 数据DataFrame
            target_column: 目标列名
        
        Returns:
            基线评估结果
        """
        # 预测基线
        baseline_values = self.predict(data, target_column)
        
        # 获取实际值
        if 'timestamp' in data.columns:
            actual_values = data[target_column].values
        else:
            actual_values = data[target_column].values
        
        # 计算节能量
        savings = baseline_values - actual_values
        
        # 计算节能百分比
        total_baseline = np.sum(baseline_values)
        total_actual = np.sum(actual_values)
        savings_percent = ((total_baseline - total_actual) / total_baseline * 100) if total_baseline > 0 else 0
        
        # 计算误差指标
        mape = np.mean(np.abs((actual_values - baseline_values) / np.maximum(actual_values, 1e-8))) * 100
        rmse = np.sqrt(mean_squared_error(actual_values, baseline_values))
        
        return BaselineResult(
            baseline_values=baseline_values,
            actual_values=actual_values,
            savings=savings,
            savings_percent=savings_percent,
            mape=mape,
            rmse=rmse
        )
    
    def save(self, path: str) -> None:
        """保存模型"""
        # 保存元数据
        meta_path = path + '.meta.json'
        with open(meta_path, 'w') as f:
            json.dump({'baseline_type': self.baseline_type.value}, f)
        
        # 保存模型
        self.model.save(path)
    
    def load(self, path: str) -> None:
        """加载模型"""
        # 加载元数据
        meta_path = path + '.meta.json'
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        self.baseline_type = BaselineType(meta['baseline_type'])
        self._create_model()
        self.model.load(path)
