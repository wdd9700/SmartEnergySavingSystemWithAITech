"""
CPU拓扑检测器单元测试

测试覆盖:
- 数据类型定义
- AMD CCD检测
- Intel大小核检测
- 主检测器功能
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from topology_types import (
    CPUVendor, CoreType, CCDType, TaskType,
    CoreInfo, CPUTopology
)
from amd_detector import AMDCCDDetector
from intel_detector import IntelHybridDetector
from topology import CPUTopologyDetector


class TestCPUVendor(unittest.TestCase):
    """测试CPU厂商枚举"""
    
    def test_vendor_values(self):
        """测试厂商枚举值"""
        self.assertEqual(CPUVendor.AMD.value, "AuthenticAMD")
        self.assertEqual(CPUVendor.INTEL.value, "GenuineIntel")
        self.assertEqual(CPUVendor.UNKNOWN.value, "Unknown")


class TestCoreType(unittest.TestCase):
    """测试核心类型枚举"""
    
    def test_core_type_values(self):
        """测试核心类型枚举值"""
        self.assertEqual(CoreType.P_CORE.value, "performance")
        self.assertEqual(CoreType.E_CORE.value, "efficient")
        self.assertEqual(CoreType.SOC_CORE.value, "soc")
        self.assertEqual(CoreType.UNKNOWN.value, "unknown")


class TestCCDType(unittest.TestCase):
    """测试CCD类型枚举"""
    
    def test_ccd_type_values(self):
        """测试CCD类型枚举值"""
        self.assertEqual(CCDType.CCD0_CACHE.value, "ccd0_cache")
        self.assertEqual(CCDType.CCD1_FREQ.value, "ccd1_freq")
        self.assertEqual(CCDType.UNKNOWN.value, "unknown")


class TestCoreInfo(unittest.TestCase):
    """测试核心信息数据类"""
    
    def test_core_info_creation(self):
        """测试创建CoreInfo对象"""
        core = CoreInfo(
            core_id=0,
            logical_processor_id=0,
            numa_node=0,
            l3_cache_id=0,
            core_type=CoreType.P_CORE,
            base_freq_mhz=3500.0,
            max_freq_mhz=5000.0
        )
        
        self.assertEqual(core.core_id, 0)
        self.assertEqual(core.logical_processor_id, 0)
        self.assertEqual(core.numa_node, 0)
        self.assertEqual(core.l3_cache_id, 0)
        self.assertEqual(core.core_type, CoreType.P_CORE)
        self.assertEqual(core.base_freq_mhz, 3500.0)
        self.assertEqual(core.max_freq_mhz, 5000.0)
    
    def test_core_info_defaults(self):
        """测试CoreInfo默认值"""
        core = CoreInfo(
            core_id=1,
            logical_processor_id=2,
            numa_node=0,
            l3_cache_id=1
        )
        
        self.assertIsNone(core.core_type)
        self.assertIsNone(core.ccd_type)
        self.assertEqual(core.base_freq_mhz, 0.0)
        self.assertEqual(core.max_freq_mhz, 0.0)


class TestCPUTopology(unittest.TestCase):
    """测试CPU拓扑数据类"""
    
    def test_topology_creation(self):
        """测试创建CPUTopology对象"""
        cores = [
            CoreInfo(core_id=0, logical_processor_id=0, numa_node=0, l3_cache_id=0),
            CoreInfo(core_id=1, logical_processor_id=1, numa_node=0, l3_cache_id=0),
        ]
        
        topology = CPUTopology(
            vendor=CPUVendor.INTEL,
            model_name="Intel Core i9-13900K",
            physical_cores=24,
            logical_cores=32,
            numa_nodes=1,
            l3_caches=1,
            cores=cores,
            p_core_count=8,
            e_core_count=16
        )
        
        self.assertEqual(topology.vendor, CPUVendor.INTEL)
        self.assertEqual(topology.model_name, "Intel Core i9-13900K")
        self.assertEqual(topology.physical_cores, 24)
        self.assertEqual(topology.logical_cores, 32)
        self.assertEqual(topology.numa_nodes, 1)
        self.assertEqual(topology.l3_caches, 1)
        self.assertEqual(len(topology.cores), 2)
        self.assertEqual(topology.p_core_count, 8)
        self.assertEqual(topology.e_core_count, 16)


class TestAMDCCDDetector(unittest.TestCase):
    """测试AMD CCD检测器"""
    
    def setUp(self):
        """测试前准备"""
        self.detector = AMDCCDDetector()
    
    def test_is_x3d_processor(self):
        """测试X3D处理器识别"""
        self.assertTrue(self.detector._is_x3d_processor("AMD Ryzen 9 7950X3D"))
        self.assertTrue(self.detector._is_x3d_processor("AMD Ryzen 9 9950X3D"))
        self.assertTrue(self.detector._is_x3d_processor("Ryzen 7 7800X3D"))
        self.assertFalse(self.detector._is_x3d_processor("AMD Ryzen 9 7950X"))
        self.assertFalse(self.detector._is_x3d_processor("Intel Core i9-13900K"))
    
    def test_infer_ccd_from_cache_single_ccd(self):
        """测试单CCD缓存推断"""
        cores = [
            CoreInfo(core_id=0, logical_processor_id=0, numa_node=0, l3_cache_id=0),
            CoreInfo(core_id=1, logical_processor_id=1, numa_node=0, l3_cache_id=0),
            CoreInfo(core_id=2, logical_processor_id=2, numa_node=0, l3_cache_id=0),
            CoreInfo(core_id=3, logical_processor_id=3, numa_node=0, l3_cache_id=0),
        ]
        
        result = self.detector._infer_ccd_from_cache(cores)
        
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["ccd0_cores"], [0, 1, 2, 3])
        self.assertEqual(result["ccd1_cores"], [])
        self.assertEqual(result["mapping"][0], CCDType.CCD0_CACHE)
        self.assertEqual(result["mapping"][3], CCDType.CCD0_CACHE)
    
    def test_infer_ccd_from_cache_dual_ccd(self):
        """测试双CCD缓存推断"""
        cores = [
            # CCD0
            CoreInfo(core_id=0, logical_processor_id=0, numa_node=0, l3_cache_id=0),
            CoreInfo(core_id=1, logical_processor_id=1, numa_node=0, l3_cache_id=0),
            CoreInfo(core_id=2, logical_processor_id=2, numa_node=0, l3_cache_id=0),
            CoreInfo(core_id=3, logical_processor_id=3, numa_node=0, l3_cache_id=0),
            # CCD1
            CoreInfo(core_id=4, logical_processor_id=4, numa_node=0, l3_cache_id=1),
            CoreInfo(core_id=5, logical_processor_id=5, numa_node=0, l3_cache_id=1),
            CoreInfo(core_id=6, logical_processor_id=6, numa_node=0, l3_cache_id=1),
            CoreInfo(core_id=7, logical_processor_id=7, numa_node=0, l3_cache_id=1),
        ]
        
        result = self.detector._infer_ccd_from_cache(cores)
        
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["ccd0_cores"], [0, 1, 2, 3])
        self.assertEqual(result["ccd1_cores"], [4, 5, 6, 7])
        self.assertEqual(result["mapping"][0], CCDType.CCD0_CACHE)
        self.assertEqual(result["mapping"][4], CCDType.CCD1_FREQ)
    
    def test_get_ccd0_cores(self):
        """测试获取CCD0核心"""
        cores = [
            CoreInfo(core_id=0, logical_processor_id=0, numa_node=0, l3_cache_id=0, ccd_type=CCDType.CCD0_CACHE),
            CoreInfo(core_id=1, logical_processor_id=1, numa_node=0, l3_cache_id=0, ccd_type=CCDType.CCD0_CACHE),
            CoreInfo(core_id=2, logical_processor_id=2, numa_node=0, l3_cache_id=1, ccd_type=CCDType.CCD1_FREQ),
        ]
        
        ccd0_cores = self.detector.get_ccd0_cores(cores)
        
        self.assertEqual(ccd0_cores, [0, 1])
    
    def test_get_ccd1_cores(self):
        """测试获取CCD1核心"""
        cores = [
            CoreInfo(core_id=0, logical_processor_id=0, numa_node=0, l3_cache_id=0, ccd_type=CCDType.CCD0_CACHE),
            CoreInfo(core_id=1, logical_processor_id=1, numa_node=0, l3_cache_id=0, ccd_type=CCDType.CCD0_CACHE),
            CoreInfo(core_id=2, logical_processor_id=2, numa_node=0, l3_cache_id=1, ccd_type=CCDType.CCD1_FREQ),
        ]
        
        ccd1_cores = self.detector.get_ccd1_cores(cores)
        
        self.assertEqual(ccd1_cores, [2])
    
    def test_recommended_affinity_interactive(self):
        """测试交互式任务亲和度推荐"""
        cores = [
            CoreInfo(core_id=0, logical_processor_id=0, numa_node=0, l3_cache_id=0, ccd_type=CCDType.CCD0_CACHE),
            CoreInfo(core_id=1, logical_processor_id=1, numa_node=0, l3_cache_id=0, ccd_type=CCDType.CCD0_CACHE),
            CoreInfo(core_id=2, logical_processor_id=2, numa_node=0, l3_cache_id=1, ccd_type=CCDType.CCD1_FREQ),
            CoreInfo(core_id=3, logical_processor_id=3, numa_node=0, l3_cache_id=1, ccd_type=CCDType.CCD1_FREQ),
        ]
        
        affinity = self.detector.get_recommended_affinity(cores, "interactive")
        
        # 交互式任务应该优先使用CCD1（高频率）
        self.assertEqual(affinity, {2, 3})
    
    def test_recommended_affinity_cache_sensitive(self):
        """测试缓存敏感任务亲和度推荐"""
        cores = [
            CoreInfo(core_id=0, logical_processor_id=0, numa_node=0, l3_cache_id=0, ccd_type=CCDType.CCD0_CACHE),
            CoreInfo(core_id=1, logical_processor_id=1, numa_node=0, l3_cache_id=0, ccd_type=CCDType.CCD0_CACHE),
            CoreInfo(core_id=2, logical_processor_id=2, numa_node=0, l3_cache_id=1, ccd_type=CCDType.CCD1_FREQ),
        ]
        
        affinity = self.detector.get_recommended_affinity(cores, "cache_sensitive")
        
        # 缓存敏感任务应该使用CCD0（大缓存）
        self.assertEqual(affinity, {0, 1})
    
    def test_estimate_ccd_frequencies_9950x3d(self):
        """测试9950X3D频率估算"""
        freqs = self.detector.estimate_ccd_frequencies("AMD Ryzen 9 9950X3D")
        
        self.assertEqual(freqs["ccd0_max"], 5250.0)
        self.assertEqual(freqs["ccd1_max"], 5700.0)


class TestIntelHybridDetector(unittest.TestCase):
    """测试Intel大小核检测器"""
    
    def setUp(self):
        """测试前准备"""
        self.detector = IntelHybridDetector()
    
    def test_parse_intel_config_13900k(self):
        """测试i9-13900K配置解析"""
        config = self.detector._parse_intel_config("I9-13900K")
        
        self.assertIsNotNone(config)
        self.assertEqual(config["p_cores"], 8)
        self.assertEqual(config["e_cores"], 16)
        self.assertEqual(config["soc_cores"], 0)
    
    def test_parse_intel_config_ultra(self):
        """测试Core Ultra配置解析"""
        config = self.detector._parse_intel_config("ULTRA 9 285K")
        
        self.assertIsNotNone(config)
        self.assertEqual(config["p_cores"], 8)
        self.assertEqual(config["e_cores"], 16)
        self.assertEqual(config["soc_cores"], 2)
    
    def test_parse_intel_config_unknown(self):
        """测试未知型号配置解析"""
        config = self.detector._parse_intel_config("UNKNOWN CPU")
        
        self.assertIsNone(config)
    
    def test_detect_from_model_name(self):
        """测试基于型号的检测"""
        cores = [
            CoreInfo(core_id=0, logical_processor_id=0, numa_node=0, l3_cache_id=0),
            CoreInfo(core_id=1, logical_processor_id=1, numa_node=0, l3_cache_id=0),
            CoreInfo(core_id=2, logical_processor_id=2, numa_node=0, l3_cache_id=0),
            CoreInfo(core_id=3, logical_processor_id=3, numa_node=0, l3_cache_id=0),
        ]
        
        result = self.detector._detect_from_model_name("I3-12100", cores)
        
        # i3-12100是4P-Core，无E-Core
        self.assertEqual(result["p_cores"], [0, 1, 2, 3])
        self.assertEqual(result["e_cores"], [])
    
    def test_get_p_cores(self):
        """测试获取P-Core"""
        cores = [
            CoreInfo(core_id=0, logical_processor_id=0, numa_node=0, l3_cache_id=0, core_type=CoreType.P_CORE),
            CoreInfo(core_id=1, logical_processor_id=1, numa_node=0, l3_cache_id=0, core_type=CoreType.P_CORE),
            CoreInfo(core_id=2, logical_processor_id=2, numa_node=0, l3_cache_id=0, core_type=CoreType.E_CORE),
        ]
        
        p_cores = self.detector.get_p_cores(cores)
        
        self.assertEqual(p_cores, [0, 1])
    
    def test_get_e_cores(self):
        """测试获取E-Core"""
        cores = [
            CoreInfo(core_id=0, logical_processor_id=0, numa_node=0, l3_cache_id=0, core_type=CoreType.P_CORE),
            CoreInfo(core_id=1, logical_processor_id=1, numa_node=0, l3_cache_id=0, core_type=CoreType.E_CORE),
            CoreInfo(core_id=2, logical_processor_id=2, numa_node=0, l3_cache_id=0, core_type=CoreType.E_CORE),
        ]
        
        e_cores = self.detector.get_e_cores(cores)
        
        self.assertEqual(e_cores, [1, 2])
    
    def test_get_soc_cores(self):
        """测试获取SoC Core"""
        cores = [
            CoreInfo(core_id=0, logical_processor_id=0, numa_node=0, l3_cache_id=0, core_type=CoreType.P_CORE),
            CoreInfo(core_id=1, logical_processor_id=1, numa_node=0, l3_cache_id=0, core_type=CoreType.SOC_CORE),
        ]
        
        soc_cores = self.detector.get_soc_cores(cores)
        
        self.assertEqual(soc_cores, [1])
    
    def test_recommended_affinity_interactive(self):
        """测试交互式任务亲和度推荐"""
        cores = [
            CoreInfo(core_id=0, logical_processor_id=0, numa_node=0, l3_cache_id=0, core_type=CoreType.P_CORE),
            CoreInfo(core_id=1, logical_processor_id=1, numa_node=0, l3_cache_id=0, core_type=CoreType.P_CORE),
            CoreInfo(core_id=2, logical_processor_id=2, numa_node=0, l3_cache_id=0, core_type=CoreType.E_CORE),
        ]
        
        affinity = self.detector.get_recommended_affinity(cores, "interactive")
        
        # 交互式任务应该优先使用P-Core
        self.assertEqual(affinity, {0, 1})
    
    def test_recommended_affinity_background(self):
        """测试后台任务亲和度推荐"""
        cores = [
            CoreInfo(core_id=0, logical_processor_id=0, numa_node=0, l3_cache_id=0, core_type=CoreType.P_CORE),
            CoreInfo(core_id=1, logical_processor_id=1, numa_node=0, l3_cache_id=0, core_type=CoreType.E_CORE),
            CoreInfo(core_id=2, logical_processor_id=2, numa_node=0, l3_cache_id=0, core_type=CoreType.E_CORE),
        ]
        
        affinity = self.detector.get_recommended_affinity(cores, "background")
        
        # 后台任务应该优先使用E-Core
        self.assertEqual(affinity, {1, 2})


class TestCPUTopologyDetector(unittest.TestCase):
    """测试CPU拓扑检测器主类"""
    
    def setUp(self):
        """测试前准备"""
        self.detector = CPUTopologyDetector()
    
    @patch('topology.CPUTopologyDetector._get_cpu_vendor')
    @patch('topology.CPUTopologyDetector._get_cpu_model_name')
    @patch('topology.CPUTopologyDetector._get_processor_topology')
    def test_detect_amd(self, mock_topology, mock_model, mock_vendor):
        """测试AMD处理器检测"""
        mock_vendor.return_value = CPUVendor.AMD
        mock_model.return_value = "AMD Ryzen 9 9950X3D"
        mock_topology.return_value = {
            "physical_cores": 16,
            "logical_cores": 32,
            "numa_nodes": 1,
            "l3_caches": 2,
            "cores": {
                0: {"logical_processors": [0, 1], "flags": 0, "efficiency_class": 0},
            },
            "numa_nodes_info": {0: [0, 1]},
            "l3_caches_info": {0: {"logical_processors": [0, 1], "size": 32768000}}
        }
        
        topology = self.detector.detect(use_cache=False)
        
        self.assertEqual(topology.vendor, CPUVendor.AMD)
        self.assertEqual(topology.model_name, "AMD Ryzen 9 9950X3D")
        self.assertTrue(topology.has_3d_vcache)
    
    @patch('topology.CPUTopologyDetector._get_cpu_vendor')
    @patch('topology.CPUTopologyDetector._get_cpu_model_name')
    @patch('topology.CPUTopologyDetector._get_processor_topology')
    def test_detect_intel(self, mock_topology, mock_model, mock_vendor):
        """测试Intel处理器检测"""
        mock_vendor.return_value = CPUVendor.INTEL
        mock_model.return_value = "Intel Core i9-13900K"
        mock_topology.return_value = {
            "physical_cores": 24,
            "logical_cores": 32,
            "numa_nodes": 1,
            "l3_caches": 1,
            "cores": {
                0: {"logical_processors": [0, 1], "flags": 0, "efficiency_class": 1},
            },
            "numa_nodes_info": {0: [0, 1]},
            "l3_caches_info": {0: {"logical_processors": [0, 1], "size": 36700160}}
        }
        
        topology = self.detector.detect(use_cache=False)
        
        self.assertEqual(topology.vendor, CPUVendor.INTEL)
        self.assertEqual(topology.model_name, "Intel Core i9-13900K")
    
    def test_cache_functionality(self):
        """测试缓存功能"""
        # 清除缓存
        self.detector.clear_cache()
        self.assertIsNone(self.detector._topology_cache)
        
        # 模拟检测结果
        mock_topology = Mock()
        mock_topology.vendor = CPUVendor.INTEL
        self.detector._topology_cache = mock_topology
        self.detector._cache_timestamp = 9999999999  # 未来时间戳
        
        # 使用缓存
        result = self.detector.detect(use_cache=True)
        self.assertEqual(result, mock_topology)
    
    def test_get_recommended_affinity_unknown_vendor(self):
        """测试未知厂商的亲和度推荐"""
        mock_topology = Mock()
        mock_topology.vendor = CPUVendor.UNKNOWN
        mock_topology.cores = [
            CoreInfo(core_id=0, logical_processor_id=0, numa_node=0, l3_cache_id=0),
            CoreInfo(core_id=1, logical_processor_id=1, numa_node=0, l3_cache_id=0),
        ]
        
        self.detector._topology_cache = mock_topology
        self.detector._cache_timestamp = 9999999999
        
        affinity = self.detector.get_recommended_affinity("interactive")
        
        self.assertEqual(affinity, {0, 1})


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_full_workflow_amd(self):
        """测试AMD完整工作流程"""
        # 创建模拟核心数据
        cores = []
        for i in range(16):
            l3_id = 0 if i < 8 else 1
            cores.append(CoreInfo(
                core_id=i,
                logical_processor_id=i,
                numa_node=0,
                l3_cache_id=l3_id
            ))
        
        detector = AMDCCDDetector()
        result = detector.detect_ccd_topology("AMD Ryzen 9 9950X3D", cores)
        
        # 验证结果
        self.assertEqual(result["ccd_count"], 2)
        self.assertTrue(result["has_3d_vcache"])
        self.assertEqual(len(result["ccd0_cores"]), 8)
        self.assertEqual(len(result["ccd1_cores"]), 8)
        
        # 验证核心类型已更新
        self.assertEqual(cores[0].ccd_type, CCDType.CCD0_CACHE)
        self.assertEqual(cores[8].ccd_type, CCDType.CCD1_FREQ)
    
    def test_full_workflow_intel(self):
        """测试Intel完整工作流程"""
        cores = []
        for i in range(24):
            cores.append(CoreInfo(
                core_id=i,
                logical_processor_id=i,
                numa_node=0,
                l3_cache_id=0
            ))
        
        detector = IntelHybridDetector()
        result = detector.detect_hybrid_topology("Intel Core i9-13900K", cores)
        
        # 验证结果
        self.assertEqual(result["p_core_count"], 8)
        self.assertEqual(result["e_core_count"], 16)
        self.assertEqual(len(result["p_cores"]), 8)
        self.assertEqual(len(result["e_cores"]), 16)


def run_tests():
    """运行所有测试"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加所有测试类
    suite.addTests(loader.loadTestsFromTestCase(TestCPUVendor))
    suite.addTests(loader.loadTestsFromTestCase(TestCoreType))
    suite.addTests(loader.loadTestsFromTestCase(TestCCDType))
    suite.addTests(loader.loadTestsFromTestCase(TestCoreInfo))
    suite.addTests(loader.loadTestsFromTestCase(TestCPUTopology))
    suite.addTests(loader.loadTestsFromTestCase(TestAMDCCDDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestIntelHybridDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestCPUTopologyDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
