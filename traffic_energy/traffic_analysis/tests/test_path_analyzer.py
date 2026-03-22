#!/usr/bin/env python3
"""
路径分析器测试模块

测试路径分析器的各项功能，包括轨迹聚类、路径-时间图生成和拥堵热点识别。
"""

import unittest
import sys
import os
from datetime import datetime, timedelta

import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from traffic_energy.traffic_analysis.path_analyzer import (
    PathAnalyzer,
    CameraTopology,
    PathSegment,
    TurnEvent
)
from traffic_energy.traffic_analysis.trajectory_clustering import (
    VehicleTrajectory,
    TrajectoryPoint
)


class TestPathAnalyzer(unittest.TestCase):
    """路径分析器测试类"""
    
    def setUp(self):
        """测试前准备"""
        topology = CameraTopology(
            camera_id="test_cam",
            position=(320, 240),
            zones={
                "entry": [(0, 0), (100, 0), (100, 100), (0, 100)],
                "exit": [(540, 380), (640, 380), (640, 480), (540, 480)]
            }
        )
        self.analyzer = PathAnalyzer(
            camera_topology=topology,
            cluster_eps=50.0,
            cluster_min_samples=3,
            time_window=3600
        )
    
    def create_sample_trajectory_points(self, num_points=10):
        """创建示例轨迹点"""
        points = []
        base_time = datetime.now()
        
        for i in range(num_points):
            x = 100 + i * 20
            y = 100 + i * 10
            timestamp = (base_time + timedelta(seconds=i)).timestamp()
            points.append(TrajectoryPoint(timestamp=timestamp, center=(x, y)))
        
        return points
    
    def create_sample_vehicle_trajectory(
        self,
        track_id: int,
        start_point: tuple = (100, 100),
        end_point: tuple = (300, 200),
        num_points: int = 10
    ) -> VehicleTrajectory:
        """创建示例车辆轨迹"""
        points = []
        entry_time = datetime.now()
        
        for i in range(num_points):
            ratio = i / (num_points - 1)
            x = start_point[0] + ratio * (end_point[0] - start_point[0])
            y = start_point[1] + ratio * (end_point[1] - start_point[1])
            
            # 添加随机噪声
            x += np.random.normal(0, 3)
            y += np.random.normal(0, 3)
            
            timestamp = (entry_time + timedelta(seconds=i * 2)).timestamp()
            points.append(TrajectoryPoint(timestamp=timestamp, center=(x, y)))
        
        exit_time = entry_time + timedelta(seconds=num_points * 2)
        
        return VehicleTrajectory(
            track_id=track_id,
            vehicle_type="car",
            power_type="fuel",
            entry_time=entry_time,
            exit_time=exit_time,
            entry_zone="entry",
            exit_zone="exit",
            path_points=points
        )
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.analyzer.topology)
        self.assertEqual(self.analyzer.topology.camera_id, "test_cam")
        self.assertEqual(len(self.analyzer.topology.zones), 2)
    
    def test_add_trajectory(self):
        """测试添加轨迹"""
        points = self.create_sample_trajectory_points()
        
        self.analyzer.add_trajectory(
            track_id=1,
            trajectory=points,
            vehicle_type="car",
            power_type="fuel",
            entry_zone="entry",
            exit_zone="exit"
        )
        
        self.assertEqual(len(self.analyzer._trajectories), 1)
        self.assertEqual(len(self.analyzer._vehicle_trajectories), 1)
    
    def test_add_vehicle_trajectory(self):
        """测试添加车辆轨迹对象"""
        traj = self.create_sample_vehicle_trajectory(1)
        
        self.analyzer.add_vehicle_trajectory(traj)
        
        self.assertEqual(len(self.analyzer._vehicle_trajectories), 1)
    
    def test_cluster(self):
        """测试轨迹聚类"""
        # 添加多条相似轨迹
        for i in range(5):
            traj = self.create_sample_vehicle_trajectory(
                i, (100, 100), (300, 200)
            )
            self.analyzer.add_vehicle_trajectory(traj)
        
        clusters = self.analyzer.cluster()
        
        # 应该至少有一个聚类
        self.assertGreaterEqual(len(clusters), 1)
        
        # 验证聚类存储
        self.assertEqual(len(self.analyzer._clusters), len(clusters))
    
    def test_generate_path_time_map(self):
        """测试生成路径-时间图"""
        # 先添加轨迹并聚类
        for i in range(5):
            traj = self.create_sample_vehicle_trajectory(i)
            self.analyzer.add_vehicle_trajectory(traj)
        
        self.analyzer.cluster()
        
        # 生成路径-时间图
        path_time_maps = self.analyzer.generate_path_time_map()
        
        # 应该有结果
        self.assertIsInstance(path_time_maps, list)
    
    def test_get_congestion_hotspots(self):
        """测试获取拥堵热点"""
        # 添加轨迹
        for i in range(5):
            traj = self.create_sample_vehicle_trajectory(i)
            self.analyzer.add_vehicle_trajectory(traj)
        
        self.analyzer.cluster()
        self.analyzer.generate_path_time_map()
        
        # 获取拥堵热点
        hotspots = self.analyzer.get_congestion_hotspots(threshold="low")
        
        # 返回应该是列表
        self.assertIsInstance(hotspots, list)
    
    def test_get_cluster_statistics(self):
        """测试获取聚类统计"""
        # 添加轨迹并聚类
        for i in range(5):
            traj = self.create_sample_vehicle_trajectory(i)
            self.analyzer.add_vehicle_trajectory(traj)
        
        self.analyzer.cluster()
        
        stats = self.analyzer.get_cluster_statistics()
        
        self.assertIn("total_clusters", stats)
        self.assertIn("total_trajectories", stats)
        self.assertEqual(stats["total_trajectories"], 5)
    
    def test_turn_detection(self):
        """测试转向检测"""
        # 创建带有转向的轨迹
        points = []
        base_time = datetime.now()
        
        # 先向右
        for i in range(5):
            timestamp = (base_time + timedelta(seconds=i)).timestamp()
            points.append(TrajectoryPoint(timestamp=timestamp, center=(100 + i*20, 100)))
        
        # 再向下（右转）
        for i in range(5):
            timestamp = (base_time + timedelta(seconds=5+i)).timestamp()
            points.append(TrajectoryPoint(timestamp=timestamp, center=(200, 100 + i*20)))
        
        self.analyzer.add_trajectory(1, points)
        
        # 检查是否检测到转向
        turn_ratio = self.analyzer.get_turn_ratio()
        self.assertIsInstance(turn_ratio, dict)
    
    def test_od_matrix(self):
        """测试OD矩阵"""
        # 创建带区域的轨迹
        zones = {
            "zone_a": [(0, 0), (150, 0), (150, 150), (0, 150)],
            "zone_b": [(450, 300), (640, 300), (640, 480), (450, 480)]
        }
        
        points1 = [
            TrajectoryPoint(timestamp=datetime.now().timestamp(), center=(50, 50)),
            TrajectoryPoint(timestamp=datetime.now().timestamp(), center=(550, 400))
        ]
        
        points2 = [
            TrajectoryPoint(timestamp=datetime.now().timestamp(), center=(550, 400)),
            TrajectoryPoint(timestamp=datetime.now().timestamp(), center=(50, 50))
        ]
        
        self.analyzer.add_trajectory(1, points1)
        self.analyzer.add_trajectory(2, points2)
        
        od_matrix = self.analyzer.get_origin_destination_matrix(zones)
        
        self.assertIsInstance(od_matrix, dict)
    
    def test_clear(self):
        """测试清除功能"""
        # 添加数据
        traj = self.create_sample_vehicle_trajectory(1)
        self.analyzer.add_vehicle_trajectory(traj)
        self.analyzer.cluster()
        
        # 验证数据存在
        self.assertGreater(len(self.analyzer._vehicle_trajectories), 0)
        
        # 清除
        self.analyzer.clear()
        
        # 验证数据已清除
        self.assertEqual(len(self.analyzer._trajectories), 0)
        self.assertEqual(len(self.analyzer._vehicle_trajectories), 0)
        self.assertEqual(len(self.analyzer._clusters), 0)
    
    def test_empty_cluster(self):
        """测试空轨迹聚类"""
        clusters = self.analyzer.cluster()
        self.assertEqual(len(clusters), 0)
    
    def test_insufficient_trajectories(self):
        """测试轨迹数量不足"""
        # 只添加2条轨迹（少于min_samples=3）
        for i in range(2):
            traj = self.create_sample_vehicle_trajectory(i)
            self.analyzer.add_vehicle_trajectory(traj)
        
        clusters = self.analyzer.cluster()
        self.assertEqual(len(clusters), 0)


class TestCameraTopology(unittest.TestCase):
    """摄像头拓扑测试类"""
    
    def test_topology_creation(self):
        """测试拓扑创建"""
        topology = CameraTopology(
            camera_id="cam_001",
            position=(320, 240),
            zones={
                "entry": [(0, 0), (100, 0), (100, 100), (0, 100)],
                "exit": [(500, 400), (600, 400), (600, 500), (500, 500)]
            }
        )
        
        self.assertEqual(topology.camera_id, "cam_001")
        self.assertEqual(topology.position, (320, 240))
        self.assertEqual(len(topology.zones), 2)


class TestPathSegment(unittest.TestCase):
    """路径段测试类"""
    
    def test_segment_creation(self):
        """测试路径段创建"""
        segment = PathSegment(
            start_point=(0, 0),
            end_point=(100, 100),
            direction=(1, 1),
            length=141.42
        )
        
        self.assertEqual(segment.start_point, (0, 0))
        self.assertEqual(segment.end_point, (100, 100))
        self.assertAlmostEqual(segment.length, 141.42, delta=0.1)


class TestTurnEvent(unittest.TestCase):
    """转向事件测试类"""
    
    def test_turn_event_creation(self):
        """测试转向事件创建"""
        event = TurnEvent(
            track_id=1,
            turn_type="right",
            angle=-90.0,
            location=(100, 100)
        )
        
        self.assertEqual(event.track_id, 1)
        self.assertEqual(event.turn_type, "right")
        self.assertAlmostEqual(event.angle, -90.0)


if __name__ == '__main__':
    unittest.main()
