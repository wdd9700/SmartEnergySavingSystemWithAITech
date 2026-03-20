"""
建筑能耗预测可视化模块

提供预测结果可视化、训练历史图表和对比分析图表。
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 尝试导入matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not installed. Plotting functionality will be limited.")


class EnergyPlotter:
    """
    能耗预测可视化器
    
    提供多种图表类型用于展示预测结果和分析。
    """
    
    def __init__(
        self,
        style: str = "seaborn-v0_8",
        figsize: Tuple[int, int] = (12, 6),
        dpi: int = 100
    ):
        """
        初始化可视化器
        
        Args:
            style: matplotlib样式
            figsize: 图表尺寸
            dpi: 图表DPI
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for plotting")
        
        self.style = style
        self.figsize = figsize
        self.dpi = dpi
        
        # 设置样式
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
    
    def plot_prediction_comparison(
        self,
        timestamps: List[datetime],
        actual: np.ndarray,
        predicted: np.ndarray,
        title: str = "能耗预测对比",
        ylabel: str = "能耗 (kW)",
        save_path: Optional[str] = None
    ) -> Figure:
        """
        绘制预测值与实际值对比图
        
        Args:
            timestamps: 时间戳列表
            actual: 实际值
            predicted: 预测值
            title: 图表标题
            ylabel: Y轴标签
            save_path: 保存路径
        
        Returns:
            matplotlib图表对象
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # 绘制实际值和预测值
        ax.plot(timestamps, actual, label='实际值', color='#2E86AB', linewidth=2)
        ax.plot(timestamps, predicted, label='预测值', color='#F24236', 
                linewidth=2, linestyle='--', alpha=0.8)
        
        # 填充误差区域
        ax.fill_between(timestamps, actual, predicted, alpha=0.2, color='gray', label='误差')
        
        # 设置标签和标题
        ax.set_xlabel('时间', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # 格式化x轴
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        plt.xticks(rotation=45, ha='right')
        
        # 图例和网格
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        title: str = "训练历史",
        save_path: Optional[str] = None
    ) -> Figure:
        """
        绘制训练历史曲线
        
        Args:
            history: 训练历史字典，包含'train_loss'和'val_loss'
            title: 图表标题
            save_path: 保存路径
        
        Returns:
            matplotlib图表对象
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        ax.plot(epochs, history['train_loss'], label='训练损失', 
                color='#2E86AB', linewidth=2)
        ax.plot(epochs, history['val_loss'], label='验证损失', 
                color='#F24236', linewidth=2, linestyle='--')
        
        ax.set_xlabel('轮次', fontsize=12)
        ax.set_ylabel('损失', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def plot_baseline_comparison(
        self,
        timestamps: List[datetime],
        actual: np.ndarray,
        baseline: np.ndarray,
        title: str = "能耗基线对比",
        ylabel: str = "能耗 (kW)",
        save_path: Optional[str] = None
    ) -> Figure:
        """
        绘制基线对比图
        
        Args:
            timestamps: 时间戳列表
            actual: 实际能耗
            baseline: 基线能耗
            title: 图表标题
            ylabel: Y轴标签
            save_path: 保存路径
        
        Returns:
            matplotlib图表对象
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1] * 1.5), dpi=self.dpi)
        
        # 上图：能耗对比
        ax1.plot(timestamps, baseline, label='基线', color='#F24236', linewidth=2)
        ax1.plot(timestamps, actual, label='实际', color='#2E86AB', linewidth=2)
        ax1.fill_between(timestamps, baseline, actual, 
                         where=(actual <= baseline), alpha=0.3, color='green', label='节能')
        ax1.fill_between(timestamps, baseline, actual, 
                         where=(actual > baseline), alpha=0.3, color='red', label='超耗')
        
        ax1.set_ylabel(ylabel, fontsize=12)
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 下图：节能量
        savings = baseline - actual
        colors = ['green' if s > 0 else 'red' for s in savings]
        ax2.bar(timestamps, savings, color=colors, alpha=0.6, width=0.02)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('时间', fontsize=12)
        ax2.set_ylabel('节能量 (kW)', fontsize=12)
        ax2.set_title('节能量分析', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 格式化x轴
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def plot_multi_step_prediction(
        self,
        historical_timestamps: List[datetime],
        historical_values: np.ndarray,
        prediction_timestamps: List[datetime],
        predictions: np.ndarray,
        confidence_intervals: Optional[np.ndarray] = None,
        title: str = "多步预测",
        ylabel: str = "能耗 (kW)",
        save_path: Optional[str] = None
    ) -> Figure:
        """
        绘制多步预测图
        
        Args:
            historical_timestamps: 历史时间戳
            historical_values: 历史值
            prediction_timestamps: 预测时间戳
            predictions: 预测值
            confidence_intervals: 置信区间 (n, 2) 数组
            title: 图表标题
            ylabel: Y轴标签
            save_path: 保存路径
        
        Returns:
            matplotlib图表对象
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # 绘制历史数据
        ax.plot(historical_timestamps, historical_values, 
                label='历史数据', color='#2E86AB', linewidth=2)
        
        # 绘制预测数据
        ax.plot(prediction_timestamps, predictions, 
                label='预测值', color='#F24236', linewidth=2, linestyle='--')
        
        # 绘制置信区间
        if confidence_intervals is not None:
            ax.fill_between(prediction_timestamps, 
                           confidence_intervals[:, 0], 
                           confidence_intervals[:, 1],
                           alpha=0.2, color='#F24236', label='置信区间')
        
        # 添加垂直分隔线
        if historical_timestamps and prediction_timestamps:
            split_time = historical_timestamps[-1]
            ax.axvline(x=split_time, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('时间', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 格式化x轴
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def plot_error_distribution(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        title: str = "误差分布",
        save_path: Optional[str] = None
    ) -> Figure:
        """
        绘制误差分布图
        
        Args:
            actual: 实际值
            predicted: 预测值
            title: 图表标题
            save_path: 保存路径
        
        Returns:
            matplotlib图表对象
        """
        errors = actual - predicted
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figsize[0] * 1.2, self.figsize[1] / 1.5), dpi=self.dpi)
        
        # 左图：误差直方图
        ax1.hist(errors, bins=30, color='#2E86AB', alpha=0.7, edgecolor='black')
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('误差', fontsize=12)
        ax1.set_ylabel('频数', fontsize=12)
        ax1.set_title(f'{title} - 直方图', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 右图：Q-Q图
        from scipy import stats
        stats.probplot(errors, dist="norm", plot=ax2)
        ax2.set_title(f'{title} - Q-Q图', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def plot_feature_importance(
        self,
        feature_names: List[str],
        importance_values: np.ndarray,
        title: str = "特征重要性",
        save_path: Optional[str] = None
    ) -> Figure:
        """
        绘制特征重要性图
        
        Args:
            feature_names: 特征名称列表
            importance_values: 特征重要性值
            title: 图表标题
            save_path: 保存路径
        
        Returns:
            matplotlib图表对象
        """
        # 排序
        indices = np.argsort(np.abs(importance_values))[::-1]
        sorted_names = [feature_names[i] for i in indices]
        sorted_values = importance_values[indices]
        
        fig, ax = plt.subplots(figsize=(self.figsize[0] / 1.5, self.figsize[1]), dpi=self.dpi)
        
        colors = ['#2E86AB' if v >= 0 else '#F24236' for v in sorted_values]
        bars = ax.barh(range(len(sorted_names)), sorted_values, color=colors, alpha=0.8)
        
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel('重要性', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, sorted_values)):
            ax.text(val, i, f' {val:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def plot_heatmap(
        self,
        data: pd.DataFrame,
        title: str = "能耗热力图",
        save_path: Optional[str] = None
    ) -> Figure:
        """
        绘制能耗热力图（按小时和星期）
        
        Args:
            data: 包含timestamp和能耗值的DataFrame
            title: 图表标题
            save_path: 保存路径
        
        Returns:
            matplotlib图表对象
        """
        df = data.copy()
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        # 找到能耗列
        value_col = None
        for col in ['hvac_power', 'energy', 'power', 'consumption']:
            if col in df.columns:
                value_col = col
                break
        
        if value_col is None:
            raise ValueError("No energy column found in data")
        
        # 创建小时和星期列
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        
        # 创建透视表
        pivot = df.pivot_table(
            values=value_col,
            index='hour',
            columns='day_of_week',
            aggfunc='mean'
        )
        
        # 重命名列
        day_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        pivot.columns = day_names
        
        fig, ax = plt.subplots(figsize=(10, 8), dpi=self.dpi)
        
        im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
        
        # 设置刻度
        ax.set_xticks(range(7))
        ax.set_xticklabels(day_names)
        ax.set_yticks(range(0, 24, 2))
        ax.set_yticklabels(range(0, 24, 2))
        
        ax.set_xlabel('星期', fontsize=12)
        ax.set_ylabel('小时', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('平均能耗 (kW)', fontsize=12)
        
        # 添加数值标注
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                text = ax.text(j, i, f'{pivot.values[i, j]:.1f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def save_figure(self, fig: Figure, path: str, **kwargs) -> None:
        """
        保存图表到文件
        
        Args:
            fig: matplotlib图表对象
            path: 保存路径
            **kwargs: 传递给savefig的参数
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        fig.savefig(path, dpi=self.dpi, bbox_inches='tight', **kwargs)
        logger.info(f"Figure saved to {path}")


# 便捷函数
def plot_prediction_comparison(
    timestamps: List[datetime],
    actual: np.ndarray,
    predicted: np.ndarray,
    **kwargs
) -> Optional[Figure]:
    """
    便捷函数：绘制预测对比图
    
    Args:
        timestamps: 时间戳列表
        actual: 实际值
        predicted: 预测值
        **kwargs: 其他参数
    
    Returns:
        matplotlib图表对象或None
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available")
        return None
    
    plotter = EnergyPlotter()
    return plotter.plot_prediction_comparison(timestamps, actual, predicted, **kwargs)


def plot_training_history(
    history: Dict[str, List[float]],
    **kwargs
) -> Optional[Figure]:
    """
    便捷函数：绘制训练历史
    
    Args:
        history: 训练历史字典
        **kwargs: 其他参数
    
    Returns:
        matplotlib图表对象或None
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available")
        return None
    
    plotter = EnergyPlotter()
    return plotter.plot_training_history(history, **kwargs)
