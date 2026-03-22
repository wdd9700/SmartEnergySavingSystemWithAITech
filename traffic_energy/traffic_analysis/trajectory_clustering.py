#!/usr/bin/env python3
"""
轨迹聚类模块

使用DBSCAN算法对车辆轨迹进行聚类分析，识别常见行驶路径。

Example:
    >>> from traffic_energy.traffic_analysis.trajectory_clustering import TrajectoryClusterer
    >>> from traffic_energy.detection.vehicle_tracker import TrajectoryPoint
    >>> clusterer = TrajectoryClusterer(eps=50.0, min_samples=5)
    >>> clusters = clusterer.cluster(trajectories)
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import math

import numpy as np

try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from shared.logger import setup_logger

logger = setup_logger("trajectory_clustering")


@dataclass
class TrajectoryPoint:
    """轨迹点数据类
    
    Attributes:
        timestamp: 时间戳
        center: 中心点坐标 (x, y)
        bbox: 边界框 [x1, y1, x2, y2] (可选)
    """
    timestamp: float
    center: Tuple[float, float]
    bbox: Optional[np.ndarray] = None


@dataclass
class VehicleTrajectory:
    """车辆轨迹数据类
    
    Attributes:
        track_id: 跟踪ID
        vehicle_type: 车辆类型 (car/bus/truck)
        power_type: 动力类型 (fuel/electric/unknown)
        entry_time: 进入时间
        exit_time: 离开时间
        entry_zone: 进入区域
        exit_zone: 离开区域
        path_points: 路径点列表
    """
    track_id: int
    vehicle_type: str = "unknown"
    power_type: str = "unknown"
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    entry_zone: Optional[str] = None
    exit_zone: Optional[str] = None
    path_points: List[TrajectoryPoint] = field(default_factory=list)
    
    @property
    def travel_time(self) -> float:
        """通行时间 (秒)"""
        if self.entry_time and self.exit_time:
            return (self.exit_time - self.entry_time).total_seconds()
        return 0.0
    
    @property
    def path_feature(self) -> np.ndarray:
        """路径特征向量 (用于聚类)
        
        特征包括：起点x、起点y、终点x、终点y、路径点数
        """
        if not self.path_points:
            return np.zeros(5)
        
        start = self.path_points[0].center
        end = self.path_points[-1].center
        return np.array([start[0], start[1], end[0], end[1], len(self.path_points)])
    
    @property
    def is_valid(self) -> bool:
        """检查轨迹是否有效（至少3个点）"""
        return len(self.path_points) >= 3
    
    def get_path_length(self) -> float:
        """计算路径总长度"""
        if len(self.path_points) < 2:
            return 0.0
        
        length = 0.0
        for i in range(1, len(self.path_points)):
            p1 = self.path_points[i-1].center
            p2 = self.path_points[i].center
            length += math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
        return length


@dataclass
class PathCluster:
    """路径聚类数据类
    
    Attributes:
        cluster_id: 聚类ID
        representative_path: 代表性路径点列表
        trajectories: 包含的轨迹列表
        avg_travel_time: 平均通行时间
        std_travel_time: 通行时间标准差
        vehicle_type_dist: 车型分布
        power_type_dist: 油电分布
    """
    cluster_id: int
    representative_path: List[Tuple[float, float]]
    trajectories: List[VehicleTrajectory]
    avg_travel_time: float = 0.0
    std_travel_time: float = 0.0
    vehicle_type_dist: Dict[str, int] = field(default_factory=dict)
    power_type_dist: Dict[str, int] = field(default_factory=dict)
    
    @property
    def trajectory_count(self) -> int:
        """轨迹数量"""
        return len(self.trajectories)
    
    def get_centroid(self) -> Tuple[float, float]:
        """获取聚类中心点"""
        if not self.representative_path:
            return (0.0, 0.0)
        
        mid_idx = len(self.representative_path) // 2
        return self.representative_path[mid_idx]


class TrajectoryClusterer:
    """轨迹聚类器
    
    使用DBSCAN算法对车辆轨迹进行聚类，识别常见行驶路径。
    
    Attributes:
        eps: 聚类半径 (像素距离)
        min_samples: 最小样本数
        clusters: 聚类结果
        
    Example:
        >>> clusterer = TrajectoryClusterer(eps=50.0, min_samples=5)
        >>> clusters = clusterer.cluster(trajectories)
        >>> for cluster_id, cluster in clusters.items():
        ...     print(f"路径 {cluster_id}: {cluster.trajectory_count} 辆车")
    """
    
    def __init__(
        self,
        eps: float = 50.0,
        min_samples: int = 5,
        metric: str = 'euclidean'
    ) -> None:
        """初始化轨迹聚类器
        
        Args:
            eps: 聚类半径 (像素)
            min_samples: 形成聚类的最小样本数
            metric: 距离度量方式
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn 未安装，将使用简化聚类算法")
        
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.clusters: Dict[int, PathCluster] = {}
        self._noise_trajectories: List[VehicleTrajectory] = []
        
        logger.info(f"初始化轨迹聚类器 (eps={eps}, min_samples={min_samples})")
    
    def cluster(
        self,
        trajectories: List[VehicleTrajectory]
    ) -> Dict[int, PathCluster]:
        """对轨迹进行聚类
        
        Args:
            trajectories: 轨迹列表
            
        Returns:
            聚类结果字典 {cluster_id: PathCluster}
        """
        # 过滤无效轨迹
        valid_trajectories = [t for t in trajectories if t.is_valid]
        
        if len(valid_trajectories) < self.min_samples:
            logger.warning(f"有效轨迹数量 ({len(valid_trajectories)}) 小于最小样本数")
            return {}
        
        logger.info(f"开始对 {len(valid_trajectories)} 条轨迹进行聚类")
        
        if SKLEARN_AVAILABLE:
            clusters = self._cluster_dbscan(valid_trajectories)
        else:
            clusters = self._cluster_simple(valid_trajectories)
        
        self.clusters = clusters
        logger.info(f"聚类完成，发现 {len(clusters)} 个路径模式")
        
        return clusters
    
    def _cluster_dbscan(
        self,
        trajectories: List[VehicleTrajectory]
    ) -> Dict[int, PathCluster]:
        """使用DBSCAN进行聚类
        
        Args:
            trajectories: 有效轨迹列表
            
        Returns:
            聚类结果
        """
        # 提取特征
        features = np.array([t.path_feature for t in trajectories])
        
        # 标准化特征
        feature_mean = np.mean(features, axis=0)
        feature_std = np.std(features, axis=0)
        feature_std[feature_std == 0] = 1  # 避免除零
        features_normalized = (features - feature_mean) / feature_std
        
        # DBSCAN聚类
        clustering = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric
        ).fit(features_normalized)
        
        # 分组
        cluster_groups: Dict[int, List[VehicleTrajectory]] = defaultdict(list)
        self._noise_trajectories = []
        
        for idx, label in enumerate(clustering.labels_):
            if label == -1:  # 噪声点
                self._noise_trajectories.append(trajectories[idx])
            else:
                cluster_groups[label].append(trajectories[idx])
        
        # 创建PathCluster对象
        result = {}
        for label, trajs in cluster_groups.items():
            result[label] = self._create_path_cluster(label, trajs)
        
        return result
    
    def _cluster_simple(
        self,
        trajectories: List[VehicleTrajectory]
    ) -> Dict[int, PathCluster]:
        """简化聚类算法（当sklearn不可用时使用）
        
        基于起点和终点的距离进行简单聚类
        
        Args:
            trajectories: 有效轨迹列表
            
        Returns:
            聚类结果
        """
        if not trajectories:
            return {}
        
        clusters: Dict[int, List[VehicleTrajectory]] = {}
        cluster_id = 0
        
        for traj in trajectories:
            assigned = False
            
            for cid, cluster_trajs in clusters.items():
                # 检查是否与现有聚类匹配
                if self._is_similar_trajectory(traj, cluster_trajs[0]):
                    cluster_trajs.append(traj)
                    assigned = True
                    break
            
            if not assigned:
                clusters[cluster_id] = [traj]
                cluster_id += 1
        
        # 过滤小聚类
        result = {}
        for label, trajs in clusters.items():
            if len(trajs) >= self.min_samples:
                result[label] = self._create_path_cluster(label, trajs)
            else:
                self._noise_trajectories.extend(trajs)
        
        return result
    
    def _is_similar_trajectory(
        self,
        traj1: VehicleTrajectory,
        traj2: VehicleTrajectory
    ) -> bool:
        """判断两条轨迹是否相似
        
        Args:
            traj1: 第一条轨迹
            traj2: 第二条轨迹
            
        Returns:
            是否相似
        """
        if not traj1.path_points or not traj2.path_points:
            return False
        
        # 比较起点和终点距离
        start1 = np.array(traj1.path_points[0].center)
        start2 = np.array(traj2.path_points[0].center)
        end1 = np.array(traj1.path_points[-1].center)
        end2 = np.array(traj2.path_points[-1].center)
        
        start_dist = np.linalg.norm(start1 - start2)
        end_dist = np.linalg.norm(end1 - end2)
        
        return start_dist < self.eps and end_dist < self.eps
    
    def _create_path_cluster(
        self,
        cluster_id: int,
        trajectories: List[VehicleTrajectory]
    ) -> PathCluster:
        """创建路径聚类对象
        
        Args:
            cluster_id: 聚类ID
            trajectories: 轨迹列表
            
        Returns:
            PathCluster对象
        """
        # 计算代表性路径
        rep_path = self._compute_representative_path(trajectories)
        
        # 计算通行时间统计
        travel_times = [t.travel_time for t in trajectories if t.travel_time > 0]
        avg_time = np.mean(travel_times) if travel_times else 0.0
        std_time = np.std(travel_times) if travel_times else 0.0
        
        # 计算类型分布
        vehicle_type_dist = self._compute_type_distribution(trajectories, "vehicle_type")
        power_type_dist = self._compute_type_distribution(trajectories, "power_type")
        
        return PathCluster(
            cluster_id=cluster_id,
            representative_path=rep_path,
            trajectories=trajectories,
            avg_travel_time=avg_time,
            std_travel_time=std_time,
            vehicle_type_dist=vehicle_type_dist,
            power_type_dist=power_type_dist
        )
    
    def _compute_representative_path(
        self,
        trajectories: List[VehicleTrajectory]
    ) -> List[Tuple[float, float]]:
        """计算代表性路径
        
        使用平均轨迹作为代表性路径
        
        Args:
            trajectories: 轨迹列表
            
        Returns:
            代表性路径点列表
        """
        if not trajectories:
            return []
        
        # 找到最长轨迹作为参考
        max_length = max(len(t.path_points) for t in trajectories)
        
        if max_length == 0:
            return []
        
        # 统一采样点数
        num_points = min(20, max_length)
        
        # 对所有轨迹进行插值，统一到相同点数
        interpolated_paths = []
        for traj in trajectories:
            if len(traj.path_points) >= 2:
                points = [p.center for p in traj.path_points]
                interp_points = self._interpolate_path(points, num_points)
                interpolated_paths.append(interp_points)
        
        if not interpolated_paths:
            return []
        
        # 计算平均路径
        avg_path = []
        for i in range(num_points):
            x_coords = [path[i][0] for path in interpolated_paths]
            y_coords = [path[i][1] for path in interpolated_paths]
            avg_path.append((np.mean(x_coords), np.mean(y_coords)))
        
        return avg_path
    
    def _interpolate_path(
        self,
        points: List[Tuple[float, float]],
        num_points: int
    ) -> List[Tuple[float, float]]:
        """对路径进行插值
        
        Args:
            points: 原始路径点
            num_points: 目标点数
            
        Returns:
            插值后的路径点
        """
        if len(points) == num_points:
            return points
        
        if len(points) < 2:
            return points * num_points if points else []
        
        # 计算累积距离
        distances = [0.0]
        for i in range(1, len(points)):
            dist = math.sqrt(
                (points[i][0] - points[i-1][0])**2 +
                (points[i][1] - points[i-1][1])**2
            )
            distances.append(distances[-1] + dist)
        
        total_dist = distances[-1]
        if total_dist == 0:
            return [points[0]] * num_points
        
        # 等距采样
        result = []
        for i in range(num_points):
            target_dist = total_dist * i / (num_points - 1)
            
            # 找到对应位置
            for j in range(len(distances) - 1):
                if distances[j] <= target_dist <= distances[j+1]:
                    # 线性插值
                    ratio = (target_dist - distances[j]) / (distances[j+1] - distances[j]) if distances[j+1] > distances[j] else 0
                    x = points[j][0] + ratio * (points[j+1][0] - points[j][0])
                    y = points[j][1] + ratio * (points[j+1][1] - points[j][1])
                    result.append((x, y))
                    break
            else:
                result.append(points[-1])
        
        return result
    
    def _compute_type_distribution(
        self,
        trajectories: List[VehicleTrajectory],
        attr_name: str
    ) -> Dict[str, int]:
        """计算类型分布
        
        Args:
            trajectories: 轨迹列表
            attr_name: 属性名 ('vehicle_type' 或 'power_type')
            
        Returns:
            类型分布字典
        """
        distribution = defaultdict(int)
        
        for traj in trajectories:
            type_value = getattr(traj, attr_name, "unknown")
            distribution[type_value] += 1
        
        return dict(distribution)
    
    def get_cluster_by_id(self, cluster_id: int) -> Optional[PathCluster]:
        """根据ID获取聚类
        
        Args:
            cluster_id: 聚类ID
            
        Returns:
            PathCluster对象或None
        """
        return self.clusters.get(cluster_id)
    
    def get_noise_trajectories(self) -> List[VehicleTrajectory]:
        """获取噪声轨迹（未分类的轨迹）
        
        Returns:
            噪声轨迹列表
        """
        return self._noise_trajectories
    
    def get_cluster_statistics(self) -> Dict[str, Any]:
        """获取聚类统计信息
        
        Returns:
            统计信息字典
        """
        if not self.clusters:
            return {
                "total_clusters": 0,
                "total_trajectories": 0,
                "noise_trajectories": len(self._noise_trajectories)
            }
        
        total_trajs = sum(c.trajectory_count for c in self.clusters.values())
        avg_cluster_size = total_trajs / len(self.clusters)
        
        return {
            "total_clusters": len(self.clusters),
            "total_trajectories": total_trajs,
            "noise_trajectories": len(self._noise_trajectories),
            "average_cluster_size": avg_cluster_size,
            "largest_cluster": max(c.trajectory_count for c in self.clusters.values()),
            "smallest_cluster": min(c.trajectory_count for c in self.clusters.values())
        }
    
    def clear(self) -> None:
        """清除所有聚类结果"""
        self.clusters.clear()
        self._noise_trajectories.clear()
