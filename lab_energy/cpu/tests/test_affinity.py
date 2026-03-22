"""
CPU亲和度管理模块单元测试

测试内容:
- affinity_types: 数据类型定义
- affinity_api: Windows API封装
- affinity_manager: 亲和度管理器
"""

import unittest
import sys
import os
import logging

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from affinity_types import (
    TaskType,
    AffinityPolicy,
    CPUSetInfo,
    ThreadAffinityInfo,
    ProcessAffinityInfo,
)

from affinity_api import WindowsAffinityAPI, get_affinity_api

from affinity_manager import CPUAffinityManager, get_affinity_manager

try:
    from topology_types import CPUTopology, CPUVendor, CoreType, CCDType
    from topology import CPUTopologyDetector
except ImportError:
    from ..topology_types import CPUTopology, CPUVendor, CoreType, CCDType
    from ..topology import CPUTopologyDetector


# 禁用日志输出
logging.disable(logging.CRITICAL)


class TestAffinityTypes(unittest.TestCase):
    """测试数据类型定义"""
    
    def test_task_type_enum(self):
        """测试TaskType枚举"""
        self.assertEqual(TaskType.INTERACTIVE.value, "interactive")
        self.assertEqual(TaskType.COMPUTE.value, "compute")
        self.assertEqual(TaskType.CACHE_SENSITIVE.value, "cache")
        self.assertEqual(TaskType.BACKGROUND.value, "background")
        self.assertEqual(TaskType.SYSTEM.value, "system")
    
    def test_affinity_policy(self):
        """测试AffinityPolicy数据类"""
        policy = AffinityPolicy(
            task_type=TaskType.INTERACTIVE,
            preferred_cores={0, 1, 2, 3},
            allowed_cores={0, 1, 2, 3, 4, 5, 6, 7},
            excluded_cores={8, 9},
            priority_boost=True
        )
        
        self.assertEqual(policy.task_type, TaskType.INTERACTIVE)
        self.assertEqual(policy.preferred_cores, {0, 1, 2, 3})
        self.assertEqual(policy.allowed_cores, {0, 1, 2, 3, 4, 5, 6, 7})
        self.assertEqual(policy.excluded_cores, {8, 9})
        self.assertTrue(policy.priority_boost)
    
    def test_cpu_set_info(self):
        """测试CPUSetInfo数据类"""
        cpu_set = CPUSetInfo(
            id=0,
            group=0,
            logical_processor_index=0,
            core_index=0,
            numa_node_index=0,
            efficiency_class=1,
            all_flags=0
        )
        
        self.assertEqual(cpu_set.id, 0)
        self.assertEqual(cpu_set.group, 0)
        self.assertEqual(cpu_set.efficiency_class, 1)
    
    def test_thread_affinity_info(self):
        """测试ThreadAffinityInfo数据类"""
        info = ThreadAffinityInfo(
            thread_id=1234,
            thread_handle=5678,
            current_mask=0xFF,
            current_group=0,
            cpu_set_ids=[0, 1, 2, 3]
        )
        
        self.assertEqual(info.thread_id, 1234)
        self.assertEqual(info.current_mask, 0xFF)
        self.assertEqual(info.cpu_set_ids, [0, 1, 2, 3])
    
    def test_process_affinity_info(self):
        """测试ProcessAffinityInfo数据类"""
        info = ProcessAffinityInfo(
            pid=1234,
            process_handle=5678,
            default_cpu_sets=[0, 1, 2, 3],
            current_mask=0xFF
        )
        
        self.assertEqual(info.pid, 1234)
        self.assertEqual(info.default_cpu_sets, [0, 1, 2, 3])
        self.assertEqual(info.current_mask, 0xFF)


class TestWindowsAffinityAPI(unittest.TestCase):
    """测试Windows API封装"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.api = WindowsAffinityAPI()
    
    def test_cpu_sets_support_check(self):
        """测试CPU Sets支持检查"""
        # 检查属性是否存在
        self.assertIsInstance(self.api.cpu_sets_supported, bool)
    
    def test_get_system_info(self):
        """测试获取系统信息"""
        info = self.api.get_system_info()
        
        # 验证基本字段
        self.assertIn('number_of_processors', info)
        self.assertIn('processor_architecture', info)
        self.assertIn('active_processor_mask', info)
        
        # 验证处理器数量合理
        self.assertGreater(info['number_of_processors'], 0)
        self.assertLessEqual(info['number_of_processors'], 256)
    
    def test_get_system_cpu_set_information(self):
        """测试获取CPU Set信息"""
        cpu_sets = self.api.get_system_cpu_set_information()
        
        # 如果不支持CPU Sets，应该返回空列表
        if not self.api.cpu_sets_supported:
            self.assertEqual(cpu_sets, [])
        else:
            # 验证返回的CPU Set信息格式
            for cpu_set in cpu_sets:
                self.assertIn('Id', cpu_set)
                self.assertIn('Group', cpu_set)
                self.assertIn('LogicalProcessorIndex', cpu_set)
                self.assertIn('EfficiencyClass', cpu_set)
    
    def test_get_current_ids(self):
        """测试获取当前进程和线程ID"""
        pid = self.api.get_current_process_id()
        tid = self.api.get_current_thread_id()
        
        # 验证ID为正数
        self.assertGreater(pid, 0)
        self.assertGreater(tid, 0)
        
        # 验证当前进程ID与os.getpid()一致
        self.assertEqual(pid, os.getpid())
    
    def test_get_process_affinity_mask(self):
        """测试获取进程亲和度掩码"""
        pid = os.getpid()
        result = self.api.get_process_affinity_mask(pid)
        
        if result is not None:
            process_mask, system_mask = result
            # 验证掩码为正数
            self.assertGreater(process_mask, 0)
            self.assertGreater(system_mask, 0)
            # 进程掩码应该是系统掩码的子集
            self.assertEqual(process_mask & system_mask, process_mask)
    
    def test_get_thread_affinity_mask(self):
        """测试获取线程亲和度掩码"""
        tid = self.api.get_current_thread_id()
        mask = self.api.get_thread_affinity_mask(tid)
        
        if mask is not None:
            # 验证掩码为正数
            self.assertGreater(mask, 0)


class TestCPUAffinityManager(unittest.TestCase):
    """测试CPU亲和度管理器"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.manager = CPUAffinityManager()
    
    def test_initialization(self):
        """测试管理器初始化"""
        # 验证拓扑信息已加载
        topology = self.manager.get_topology()
        self.assertIsNotNone(topology)
        
        # 验证CPU Set信息已加载
        cpu_sets = self.manager.get_cpu_set_info()
        self.assertIsInstance(cpu_sets, list)
    
    def test_is_cpu_sets_supported(self):
        """测试CPU Sets支持检查"""
        supported = self.manager.is_cpu_sets_supported()
        self.assertIsInstance(supported, bool)
    
    def test_get_recommended_cores(self):
        """测试获取推荐核心"""
        for task_type in TaskType:
            cores = self.manager.get_recommended_cores(task_type)
            self.assertIsInstance(cores, set)
            # 每个任务类型应该有推荐的核心
            self.assertGreater(len(cores), 0)
    
    def test_get_topology(self):
        """测试获取拓扑信息"""
        topology = self.manager.get_topology()
        
        self.assertIsNotNone(topology)
        self.assertIsInstance(topology.vendor, CPUVendor)
        self.assertGreater(len(topology.model_name), 0)
        self.assertGreater(topology.physical_cores, 0)
        self.assertGreater(topology.logical_cores, 0)
    
    def test_get_process_affinity_info(self):
        """测试获取进程亲和度信息"""
        pid = os.getpid()
        info = self.manager.get_process_affinity_info(pid)
        
        if info is not None:
            self.assertEqual(info.pid, pid)
            self.assertIsNotNone(info.current_mask)
    
    def test_apply_policy_with_empty_cores(self):
        """测试应用空核心策略"""
        pid = os.getpid()
        
        # 创建一个没有可用核心的策略
        policy = AffinityPolicy(
            task_type=TaskType.BACKGROUND,
            preferred_cores=set(),
            allowed_cores=set(),
            excluded_cores=set(range(256)),  # 排除所有可能的核心
            priority_boost=False
        )
        
        # 应该返回False，因为没有可用核心
        result = self.manager.apply_policy(pid, policy)
        self.assertFalse(result)
    
    def test_set_process_affinity_with_empty_list(self):
        """测试设置空CPU Set列表"""
        pid = os.getpid()
        result = self.manager.set_process_affinity(pid, [])
        self.assertFalse(result)
    
    def test_set_thread_affinity_with_empty_list(self):
        """测试设置空CPU Set列表到线程"""
        tid = self.manager._api.get_current_thread_id()
        result = self.manager.set_thread_affinity(tid, [])
        self.assertFalse(result)


class TestAffinityIntegration(unittest.TestCase):
    """集成测试"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.detector = CPUTopologyDetector()
        cls.manager = CPUAffinityManager(cls.detector)
    
    def test_manager_with_detector(self):
        """测试管理器与检测器集成"""
        topology = self.manager.get_topology()
        
        # 验证拓扑信息正确
        self.assertIsNotNone(topology)
        self.assertGreater(len(topology.cores), 0)
    
    def test_recommended_cores_match_topology(self):
        """测试推荐核心与拓扑信息匹配"""
        topology = self.manager.get_topology()
        all_cores = {c.logical_processor_id for c in topology.cores}
        
        # 验证推荐的核心都在拓扑中
        for task_type in TaskType:
            recommended = self.manager.get_recommended_cores(task_type)
            if recommended:
                # 推荐的核心应该是所有核心的子集
                self.assertTrue(recommended.issubset(all_cores) or recommended == all_cores)


class TestGlobalInstances(unittest.TestCase):
    """测试全局实例"""
    
    def test_get_affinity_api_singleton(self):
        """测试affinity_api单例"""
        api1 = get_affinity_api()
        api2 = get_affinity_api()
        self.assertIs(api1, api2)
    
    def test_get_affinity_manager_singleton(self):
        """测试affinity_manager单例"""
        manager1 = get_affinity_manager()
        manager2 = get_affinity_manager()
        self.assertIs(manager1, manager2)


class TestErrorHandling(unittest.TestCase):
    """测试错误处理"""
    
    def setUp(self):
        """测试初始化"""
        self.api = WindowsAffinityAPI()
        self.manager = CPUAffinityManager()
    
    def test_invalid_pid(self):
        """测试无效进程ID"""
        # 使用一个不太可能存在的PID
        invalid_pid = 999999
        
        result = self.api.get_process_affinity_mask(invalid_pid)
        self.assertIsNone(result)
        
        result = self.api.get_process_default_cpu_sets(invalid_pid)
        self.assertIsNone(result)
    
    def test_invalid_tid(self):
        """测试无效线程ID"""
        # 使用一个不太可能存在的TID
        invalid_tid = 999999
        
        result = self.api.get_thread_affinity_mask(invalid_tid)
        self.assertIsNone(result)
        
        result = self.api.get_thread_selected_cpu_sets(invalid_tid)
        self.assertIsNone(result)


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestAffinityTypes))
    suite.addTests(loader.loadTestsFromTestCase(TestWindowsAffinityAPI))
    suite.addTests(loader.loadTestsFromTestCase(TestCPUAffinityManager))
    suite.addTests(loader.loadTestsFromTestCase(TestAffinityIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestGlobalInstances))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorHandling))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
