#!/usr/bin/env python3
"""
轨迹聚类测试模块

测试轨迹聚类算法的正确性和性能。
"""

import unittest
import sys
import os
from datetime import datetime, timedelta
from typing import List

import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from traffic_energy.traffic_analysis.trajectory_clustering import (
    TrajectoryClusterer,
    VehicleTrajectory,
    TrajectoryPoint,
    PathCluster
)


class TestTrajectoryClustering(unittest.TestCase):
    """轨迹聚类测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.clusterer = TrajectoryClusterer(eps=50.0, min_samples=3)
    
    def create_sample_trajectory(
        self,
        track_id: int,
        start_point: tuple,
        end_point: tuple,
        num_points: int = 10,
        vehicle_type: str = "car",
        power_type: str = "fuel"
    ) -> VehicleTrajectory:
        """创建示例轨迹"""
        points = []
        entry_time = datetime.now()
        
        for i in range(num_points):
            ratio = i / (num_points - 1)
            x = start_point[0] + ratio * (end_point[0] - start_point[0])
            y = start_point[1] + ratio * (end_point[1] - start_point[1])
            
            # 添加一些随机噪声
            x += np.random.normal(0, 5)
            y += np.random.normal(0, 5)
            
            timestamp = entry_time.timestamp() + i * 0.5
            points.append(TrajectoryPoint(timestamp=timestamp, center=(x, y)))
        
        exit_time = entry_time + timedelta(seconds=num_points * 0.5)
        
        return VehicleTrajectory(
            track_id=track_id,
            vehicle_type=vehicle_type,
            power_type=power_type,
            entry_time=entry_time,
            exit_time=exit_time,
            path_points=points
        )
    
    def test_trajectory_validity(self):
        """测试轨迹有效性检查"""
        # 有效轨迹
        valid_traj = self.create_sample_trajectory(1, (100, 100), (200, 200))
        self.assertTrue(valid_traj.is_valid)
        
        # 无效轨迹（点数不足）
        invalid_traj = VehicleTrajectory(track_id=2, path_points=[])
        self.assertFalse(invalid_traj.is_valid)
        
        # 边界情况（刚好3个点）
        boundary_traj = self.create_sample_trajectory(3, (100, 100), (200, 200), num_points=3)
        self.assertTrue(boundary_traj.is_valid)
    
    def test_path_feature(self):
        """测试路径特征提取"""
        traj = self.create_sample_trajectory(1, (100, 100), (200, 200))
        feature = traj.path_feature
        
        self.assertEqual(len(feature), 5)
        self.assertAlmostEqual(feature[0], 100, delta=10)  # 起点x
        self.assertAlmostEqual(feature[1], 100, delta=10)  # 起点y
        self.assertAlmostEqual(feature[2], 200, delta=10)  # 终点x
        self.assertAlmostEqual(feature[3], 200, delta=10)  # 终点y
        self.assertEqual(feature[4], 10)  # 路径点数
    
    def test_clustering_basic(self):
        """测试基本聚类功能"""
        # 创建两条相似路径（应该聚为一类）
        trajectories = []
        for i in range(5):
            traj = self.create_sample_trajectory(
                i, (100, 100), (200, 200),
                vehicle_type="car" if i % 2 == 0 else "bus"
            )
            trajectories.append(traj)
        
        clusters = self.clusterer.cluster(trajectories)
        
        # 应该至少有一个聚类
        self.assertGreaterEqual(len(clusters), 1)
        
        # 验证聚类包含轨迹
        for cluster_id, cluster in clusters.items():
            self.assertGreater(cluster.trajectory_count, 0)
            self.assertIsNotNone(cluster.representative_path)
    
    def test_clustering_multiple_paths(self):
        """测试多路径聚类"""
        trajectories = []
        
        # 创建第一组路径（左上到右下）
        for i in range(5):
            traj = self.create_sample_trajectory(i, (100, 100), (300, 300))
            trajectories.append(traj)
        
        # 创建第二组路径（左下到右上）- 距离较远
        for i in range(5, 10):
            traj = self.create_sample_trajectory(i, (100, 400), (300, 100))
            trajectories.append(traj)
        
        clusters = self.clusterer.cluster(trajectories)
        
        # 应该识别出两个不同的路径模式
        self.assertGreaterEqual(len(clusters), 1)
        
        # 总轨迹数应该等于输入轨迹数（减去噪声）
        total_clustered = sum(c.trajectory_count for c in clusters.values())
        self.assertLessEqual(total_clustered, len(trajectories))
    
    def test_representative_path(self):
        """测试代表性路径计算"""
        trajectories = []
        for i in range(5):
            traj = self.create_sample_trajectory(i, (100, 100), (200, 200))
            trajectories.append(traj)
        
        clusters = self.clusterer.cluster(trajectories)
        
        for cluster in clusters.values():
            rep_path = cluster.representative_path
            self.assertGreater(len(rep_path), 0)
            
            # 代表性路径应该在起点和终点附近
            first_point = rep_path[0]
            last_point = rep_path[-1]
            
            # 检查起点
            self.assertAlmostEqual(first_point[0], 100, delta=30)
            self.assertAlmostEqual(first_point[1], 100, delta=30)
            
            # 检查终点
            self.assertAlmostEqual(last_point[0], 200, delta=30)
            self.assertAlmostEqual(last_point[1], 200, delta=30)
    
    def test_type_distribution(self):
        """测试类型分布计算"""
        trajectories = []
        
        # 创建不同类型和动力的轨迹
        for i in range(3):
            traj = self.create_sample_trajectory(i, (100, 100), (200, 200), vehicle_type="car", power_type="fuel")
            trajectories.append(traj)
        
        for i in range(3, 5):
            traj = self.create_sample_trajectory(i, (100, 100), (200, 200), vehicle_type="bus", power_type="electric")
            trajectories.append(traj)
        
        clusters = self.clusterer.cluster(trajectories)
        
        for cluster in clusters.values():
            # 验证车型分布
            self.assertIn("car", cluster.vehicle_type_dist)
            self.assertIn("bus", cluster.vehicle_type_dist)
            
            # 验证油电分布
            self.assertIn("fuel", cluster.power_type_dist)
            self.assertIn("electric", cluster.power_type_dist)
    
    def test_noise_detection(self):
        """测试噪声轨迹检测"""
        trajectories = []
        
        # 创建一组相似路径
        for i in range(5):
            traj = self.create_sample_trajectory(i, (100, 100), (200, 200))
            trajectories.append(traj)
        
        # 创建一个异常路径（远离其他路径）
        noise_traj = self.create_sample_trajectory(99, (500, 500), (600, 600))
        trajectories.append(noise_traj)
        
        clusters = self.clusterer.cluster(trajectories)
        
        # 噪声轨迹应该被识别
        noise_trajectories = self.clusterer.get_noise_trajectories()
        self.assertGreaterEqual(len(noise_trajectories), 0)
    
    def test_cluster_statistics(self):
        """测试聚类统计信息"""
        trajectories = []
        for i in range(10):
            traj = self.create_sample_trajectory(i, (100, 100), (200, 200))
            trajectories.append(traj)
        
        self.clusterer.cluster(trajectories)
        stats = self.clusterer.get_cluster_statistics()
        
        self.assertIn("total_clusters", stats)
        self.assertIn("total_trajectories", stats)
        self.assertIn("noise_trajectories", stats)
        
        self.assertGreaterEqual(stats["total_clusters"], 1)
        self.assertEqual(stats["total_trajectories"], 10)
    
    def test_empty_trajectories(self):
        """测试空轨迹列表"""
        clusters = self.clusterer.cluster([])
        self.assertEqual(len(clusters), 0)
    
    def test_insufficient_trajectories(self):
        """测试轨迹数量不足"""
        trajectories = [
            self.create_sample_trajectory(1, (100, 100), (200, 200)),
            self.create_sample_trajectory(2, (100, 100), (200, 200))
        ]
        
        # min_samples=3，只有2条轨迹
        clusters = self.clusterer.cluster(trajectories)
        self.assertEqual(len(clusters), 0)
    
    def test_clear(self):
        """测试清除功能"""
        trajectories = []
        for i in range(5):
            traj = self.create_sample_trajectory(i, (100, 100), (200, 200))
            trajectories.append(traj)
        
        self.clusterer.cluster(trajectories)
        self.assertGreater(len(self.clusterer.clusters), 0)
        
        self.clusterer.clear()
        self.assertEqual(len(self.clusterer.clusters), 0)
        self.assertEqual(len(self.clusterer.get_noise_trajectories()), 0)


class TestTrajectoryPoint(unittest.TestCase):
    """轨迹点测试类"""
    
    def test_trajectory_point_creation(self):
        """测试轨迹点创建"""
        point = TrajectoryPoint(
            timestamp=datetime.now().timestamp(),
            center=(100.0, 200.0),
            bbox=np.array([50, 150, 150, 250])
        )
        
        self.assertEqual(point.center, (100.0, 200.0))
        self.assertIsNotNone(point.bbox)


class TestVehicleTrajectory(unittest.TestCase):
    """车辆轨迹测试类"""
    
    def test_travel_time_calculation(self):
        """测试通行时间计算"""
        entry_time = datetime.now()
        exit_time = entry_time + timedelta(seconds=60)
        
        points = [
            TrajectoryPoint(timestamp=entry_time.timestamp(), center=(0, 0)),
            TrajectoryPoint(timestamp=entry_time.timestamp() + 30, center=(50, 50)),
            TrajectoryPoint(timestamp=exit_time.timestamp(), center=(100, 100))
        ]
        
        traj = VehicleTrajectory(
            track_id=1,
            entry_time=entry_time,
            exit_time=exit_time,
            path_points=points
        )
        
        self.assertAlmostEqual(traj.travel_time, 60.0, delta=1.0)
    
    def test_path_length_calculation(self):
        """测试路径长度计算"""
        points = [
            TrajectoryPoint(timestamp=0, center=(0, 0)),
            TrajectoryPoint(timestamp=1, center=(0, 100)),
            TrajectoryPoint(timestamp=2, center=(100, 100))
        ]
        
        traj = VehicleTrajectory(
            track_id=1,
            path_points=points
        )
        
        length = traj.get_path_length()
        expected_length = 100 + 100  # L形路径
        self.assertAlmostEqual(length, expected_length, delta=10)


if __name__ == '__main__':
    unittest.main()
