"""
建筑能耗预测模型

基于PyTorch的LSTM和神经网络模型，用于预测建筑热负荷和能耗。
支持多步预测（1-24小时），集成天气数据。
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """预测结果数据结构"""
    predictions: np.ndarray          # 预测值
    timestamps: List[datetime]       # 预测时间戳
    confidence_intervals: Optional[np.ndarray] = None  # 置信区间
    mape: Optional[float] = None     # 平均绝对百分比误差
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'predictions': self.predictions.tolist(),
            'timestamps': [t.isoformat() for t in self.timestamps],
            'confidence_intervals': self.confidence_intervals.tolist() if self.confidence_intervals is not None else None,
            'mape': self.mape
        }


class BaseRNNModel(nn.Module):
    """RNN模型基类
    
    提取LSTM和GRU模型的公共组件，减少代码重复。
    子类只需实现特定的RNN层创建逻辑。
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2
    ):
        """
        初始化RNN基类模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层维度
            num_layers: RNN层数
            output_size: 输出维度
            dropout: Dropout概率
        """
        super(BaseRNNModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout_rate = dropout
        
        # 子类需要设置 self.rnn
        self.rnn = None
        
        # 公共层
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def _create_rnn_layer(self, rnn_type: str) -> nn.Module:
        """创建RNN层，由子类调用
        
        Args:
            rnn_type: RNN类型 ('LSTM' 或 'GRU')
            
        Returns:
            RNN层模块
        """
        rnn_class = nn.LSTM if rnn_type == 'LSTM' else nn.GRU
        return rnn_class(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, seq_len, input_size)
        
        Returns:
            输出张量 (batch_size, output_size)
        """
        raise NotImplementedError("子类必须实现forward方法")
    
    def _extract_last_hidden(self, hidden_state) -> torch.Tensor:
        """提取最后一个时间步的隐藏状态
        
        Args:
            hidden_state: RNN返回的隐藏状态
            
        Returns:
            最后一个时间步的隐藏状态
        """
        # LSTM返回 (hidden, cell)，GRU返回 hidden
        if isinstance(hidden_state, tuple):
            hidden = hidden_state[0]
        else:
            hidden = hidden_state
        return hidden[-1]


class LSTMModel(BaseRNNModel):
    """LSTM预测模型"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2
    ):
        """
        初始化LSTM模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层维度
            num_layers: LSTM层数
            output_size: 输出维度
            dropout: Dropout概率
        """
        super(LSTMModel, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout
        )
        self.rnn = self._create_rnn_layer('LSTM')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, seq_len, input_size)
        
        Returns:
            输出张量 (batch_size, output_size)
        """
        # LSTM层
        lstm_out, (hidden, cell) = self.rnn(x)
        
        # 取最后一个时间步的隐藏状态
        out = self.dropout(hidden[-1])
        out = self.fc(out)
        
        return out


class GRUModel(BaseRNNModel):
    """GRU预测模型"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2
    ):
        """
        初始化GRU模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层维度
            num_layers: GRU层数
            output_size: 输出维度
            dropout: Dropout概率
        """
        super(GRUModel, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout
        )
        self.rnn = self._create_rnn_layer('GRU')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, seq_len, input_size)
        
        Returns:
            输出张量 (batch_size, output_size)
        """
        gru_out, hidden = self.rnn(x)
        out = self.dropout(hidden[-1])
        out = self.fc(out)
        return out


class MLPModel(nn.Module):
    """多层感知机预测模型"""
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = None,
        output_size: int = 1,
        dropout: float = 0.2
    ):
        """
        初始化MLP模型
        
        Args:
            input_size: 输入特征维度
            hidden_sizes: 隐藏层维度列表
            output_size: 输出维度
            dropout: Dropout概率
        """
        super(MLPModel, self).__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [128, 64, 32]
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, seq_len, input_size)
        
        Returns:
            输出张量 (batch_size, output_size)
        """
        # 展平时间维度
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.network(x)


class TimeSeriesDataset(Dataset):
    """时间序列数据集"""
    
    def __init__(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        seq_length: int = 24
    ):
        """
        初始化数据集
        
        Args:
            data: 特征数据 (n_samples, n_features)
            targets: 目标数据 (n_samples,)
            seq_length: 序列长度
        """
        self.data = data
        self.targets = targets
        self.seq_length = seq_length
    
    def __len__(self) -> int:
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取数据样本"""
        x = self.data[idx:idx + self.seq_length]
        y = self.targets[idx + self.seq_length]
        return torch.FloatTensor(x), torch.FloatTensor([y])


class EnergyPredictor:
    """
    建筑能耗预测器
    
    支持LSTM、GRU和MLP模型，用于预测建筑热负荷和能耗。
    集成天气数据，支持多步预测。
    
    Attributes:
        model_type: 模型类型 ("lstm", "gru", "mlp")
        seq_length: 序列长度（小时）
        device: 计算设备
        model: 神经网络模型
        scaler: 特征缩放器
        target_scaler: 目标值缩放器
    """
    
    FEATURE_COLUMNS = [
        'outdoor_temp', 'indoor_temp', 'indoor_humidity',
        'solar_radiation', 'occupancy', 'hour', 'day_of_week',
        'is_holiday', 'wind_speed', 'cloud_cover'
    ]
    
    def __init__(
        self,
        model_type: str = "lstm",
        seq_length: int = 24,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        device: Optional[str] = None
    ):
        """
        初始化能耗预测器
        
        Args:
            model_type: 模型类型 ("lstm", "gru", "mlp")
            seq_length: 序列长度（小时）
            hidden_size: 隐藏层维度
            num_layers: 网络层数
            dropout: Dropout概率
            learning_rate: 学习率
            device: 计算设备 ("cpu", "cuda", None=自动选择)
        """
        self.model_type = model_type.lower()
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        
        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # 初始化模型
        self.model = None
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        # 训练历史
        self.training_history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': []
        }
        
        logger.info(f"EnergyPredictor initialized with {model_type} model on {self.device}")
    
    def _build_model(self, input_size: int, output_size: int = 1) -> nn.Module:
        """
        构建神经网络模型
        
        Args:
            input_size: 输入特征维度
            output_size: 输出维度
        
        Returns:
            神经网络模型
        """
        if self.model_type == "lstm":
            model = LSTMModel(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                output_size=output_size,
                dropout=self.dropout
            )
        elif self.model_type == "gru":
            model = GRUModel(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                output_size=output_size,
                dropout=self.dropout
            )
        elif self.model_type == "mlp":
            model = MLPModel(
                input_size=input_size * self.seq_length,
                hidden_sizes=[self.hidden_size, self.hidden_size // 2, self.hidden_size // 4],
                output_size=output_size,
                dropout=self.dropout
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return model.to(self.device)
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        准备特征数据
        
        Args:
            data: 原始数据DataFrame
        
        Returns:
            特征工程后的DataFrame
        """
        df = data.copy()
        
        # 确保时间索引
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        # 时间特征
        if 'hour' not in df.columns:
            df['hour'] = df.index.hour
        if 'day_of_week' not in df.columns:
            df['day_of_week'] = df.index.dayofweek
        if 'is_holiday' not in df.columns:
            df['is_holiday'] = (df.index.dayofweek >= 5).astype(float)
        
        # 周期性编码
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # 滞后特征
        if 'outdoor_temp' in df.columns:
            df['outdoor_temp_lag1'] = df['outdoor_temp'].shift(1)
            df['outdoor_temp_lag24'] = df['outdoor_temp'].shift(24)
        
        # 滚动统计特征
        if 'hvac_power' in df.columns:
            df['hvac_power_roll_mean_24h'] = df['hvac_power'].rolling(24).mean()
            df['hvac_power_roll_std_24h'] = df['hvac_power'].rolling(24).std()
        
        # 填充缺失值
        df.fillna(method='ffill', inplace=True)
        df.fillna(0, inplace=True)
        
        return df
    
    def train(
        self,
        data: pd.DataFrame,
        target_column: str = "hvac_power",
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        patience: int = 10,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        训练预测模型
        
        Args:
            data: 训练数据DataFrame
            target_column: 目标列名
            epochs: 训练轮数
            batch_size: 批次大小
            validation_split: 验证集比例
            patience: 早停耐心值
            verbose: 是否打印训练信息
        
        Returns:
            训练历史字典
        """
        # 特征工程
        df = self._prepare_features(data)
        
        # 确定特征列
        feature_cols = [col for col in df.columns if col != target_column]
        
        # 提取数据
        X = df[feature_cols].values
        y = df[target_column].values
        
        # 数据缩放
        X_scaled = self.scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        # 划分训练集和验证集
        split_idx = int(len(X_scaled) * (1 - validation_split))
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y_scaled[:split_idx], y_scaled[split_idx:]
        
        # 创建数据集
        train_dataset = TimeSeriesDataset(X_train, y_train, self.seq_length)
        val_dataset = TimeSeriesDataset(X_val, y_val, self.seq_length)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 构建模型
        if self.model is None:
            self.model = self._build_model(input_size=X.shape[1])
        
        # 损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=patience // 2
        )
        
        # 训练循环
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_losses = []
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            
            # 验证阶段
            self.model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_losses.append(loss.item())
            
            avg_val_loss = np.mean(val_losses)
            
            # 记录历史
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(avg_val_loss)
            
            # 学习率调整
            scheduler.step(avg_val_loss)
            
            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch [{epoch+1}/{epochs}] "
                    f"Train Loss: {avg_train_loss:.6f}, "
                    f"Val Loss: {avg_val_loss:.6f}"
                )
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        logger.info("Training completed")
        return self.training_history
    
    def predict(
        self,
        data: pd.DataFrame,
        horizon: int = 24,
        target_column: str = "hvac_power"
    ) -> PredictionResult:
        """
        多步预测
        
        Args:
            data: 历史数据DataFrame
            horizon: 预测步数（小时）
            target_column: 目标列名
        
        Returns:
            预测结果对象
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        self.model.eval()
        
        # 特征工程
        df = self._prepare_features(data)
        feature_cols = [col for col in df.columns if col != target_column]
        
        # 获取最后seq_length个时间步的数据
        X = df[feature_cols].values
        X_scaled = self.scaler.transform(X)
        
        predictions = []
        current_sequence = X_scaled[-self.seq_length:].copy()
        
        # 找到目标特征在特征列中的索引
        # 注意：训练时目标列被排除在特征外，但特征工程可能添加了相关特征
        # 这里我们假设目标特征对应原始数据中的某个列，需要找到其索引
        target_feature_idx = None
        if target_column in feature_cols:
            target_feature_idx = feature_cols.index(target_column)
        
        with torch.no_grad():
            for i in range(horizon):
                # 准备输入
                x_input = torch.FloatTensor(current_sequence).unsqueeze(0).to(self.device)
                
                # 预测
                pred = self.model(x_input).cpu().numpy()[0, 0]
                predictions.append(pred)
                
                # 滑动窗口更新
                # 1. 获取当前序列的最后一步作为新时间步的基础
                new_step = current_sequence[-1].copy()
                
                # 2. 如果找到了目标特征索引，更新预测值（使用缩放后的值）
                if target_feature_idx is not None:
                    # 将预测值转换回缩放后的特征空间
                    # 使用target_scaler的均值和标准差进行转换
                    if hasattr(self.target_scaler, 'mean_') and hasattr(self.target_scaler, 'scale_'):
                        pred_scaled = (pred - self.target_scaler.mean_[0]) / self.target_scaler.scale_[0]
                        new_step[target_feature_idx] = pred_scaled
                
                # 3. 滑动窗口：移除第一步，添加更新后的新时间步
                current_sequence = np.vstack([current_sequence[1:], new_step])
        
        # 反缩放预测值
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.target_scaler.inverse_transform(predictions).flatten()
        
        # 生成时间戳
        last_timestamp = df.index[-1]
        timestamps = [last_timestamp + timedelta(hours=i+1) for i in range(horizon)]
        
        return PredictionResult(
            predictions=predictions,
            timestamps=timestamps,
            mape=None
        )
    
    def evaluate(
        self,
        test_data: pd.DataFrame,
        target_column: str = "hvac_power"
    ) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            test_data: 测试数据DataFrame
            target_column: 目标列名
        
        Returns:
            评估指标字典
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        # 特征工程
        df = self._prepare_features(test_data)
        feature_cols = [col for col in df.columns if col != target_column]
        
        X = df[feature_cols].values
        y_true = df[target_column].values
        
        X_scaled = self.scaler.transform(X)
        
        # 创建数据集
        dataset = TimeSeriesDataset(X_scaled, y_true, self.seq_length)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # 预测
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x).cpu().numpy()
                predictions.extend(outputs.flatten())
                actuals.extend(batch_y.numpy().flatten())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # 反缩放
        predictions = self.target_scaler.inverse_transform(
            predictions.reshape(-1, 1)
        ).flatten()
        
        # 计算指标
        mae = mean_absolute_error(actuals, predictions)
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        
        # MAPE计算（避免除零）
        mask = actuals != 0
        mape = np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask])) * 100
        
        # R²
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        metrics = {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2': float(r2)
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    def save(self, path: str) -> None:
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        if self.model is None:
            raise RuntimeError("No model to save")
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'seq_length': self.seq_length,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'scaler_mean': self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else None,
            'scaler_scale': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None,
            'target_scaler_mean': self.target_scaler.mean_.tolist() if hasattr(self.target_scaler, 'mean_') else None,
            'target_scaler_scale': self.target_scaler.scale_.tolist() if hasattr(self.target_scaler, 'scale_') else None,
            'training_history': self.training_history
        }
        
        torch.save(save_dict, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        # 恢复配置
        self.model_type = checkpoint['model_type']
        self.seq_length = checkpoint['seq_length']
        self.hidden_size = checkpoint['hidden_size']
        self.num_layers = checkpoint['num_layers']
        self.dropout = checkpoint['dropout']
        self.learning_rate = checkpoint['learning_rate']
        self.training_history = checkpoint.get('training_history', {'train_loss': [], 'val_loss': []})
        
        # 恢复缩放器
        if checkpoint['scaler_mean'] is not None:
            self.scaler.mean_ = np.array(checkpoint['scaler_mean'])
            self.scaler.scale_ = np.array(checkpoint['scaler_scale'])
        
        if checkpoint['target_scaler_mean'] is not None:
            self.target_scaler.mean_ = np.array(checkpoint['target_scaler_mean'])
            self.target_scaler.scale_ = np.array(checkpoint['target_scaler_scale'])
        
        # 重建模型
        # 需要从保存的状态推断输入大小
        input_size = len(checkpoint['scaler_mean']) if checkpoint['scaler_mean'] else 10
        self.model = self._build_model(input_size=input_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Model loaded from {path}")
