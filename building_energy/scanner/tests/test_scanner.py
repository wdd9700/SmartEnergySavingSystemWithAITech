"""
进程扫描器单元测试

测试范围:
    - ProcessScanner 核心功能
    - WhitelistManager 白名单管理
    - ProcessInfo 数据类
    - 配置加载

运行测试:
    pytest building_energy/scanner/tests/test_scanner.py -v
    
或者直接运行:
    python building_energy/scanner/tests/test_scanner.py
"""

import os
import sys
import time
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import importlib.util

import psutil
import yaml

# 直接加载被测试模块（绕过building_energy的__init__.py）
def load_module_from_path(module_name, file_path):
    """从文件路径加载模块"""
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# 获取scanner目录路径
SCANNER_DIR = Path(__file__).parent.parent

# 加载模块
process_scanner = load_module_from_path('test_process_scanner', SCANNER_DIR / 'process_scanner.py')
whitelist = load_module_from_path('test_whitelist', SCANNER_DIR / 'whitelist.py')

# 导入类
ProcessScanner = process_scanner.ProcessScanner
ProcessInfo = process_scanner.ProcessInfo
ProcessType = process_scanner.ProcessType
get_scanner = process_scanner.get_scanner
reset_scanner = process_scanner.reset_scanner
WhitelistManager = whitelist.WhitelistManager
WhitelistConfig = whitelist.WhitelistConfig

# 简化版的辅助函数
def create_scanner_from_config(config_path=None, **kwargs):
    """从配置创建扫描器"""
    scanner_config = {
        'long_running_threshold': 30,
        'scan_interval_minutes': 10,
    }
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                if data and 'scanner' in data:
                    scanner_config.update(data['scanner'])
        except Exception:
            pass
    
    scanner_config.update(kwargs)
    
    return ProcessScanner(
        long_running_threshold=scanner_config.get('long_running_threshold', 30),
        scan_interval_minutes=scanner_config.get('scan_interval_minutes', 10),
    )

def quick_check():
    """快速检查"""
    scanner = get_scanner()
    return scanner.should_prevent_shutdown()


class TestProcessInfo(unittest.TestCase):
    """测试 ProcessInfo 数据类"""
    
    def test_create_process_info(self):
        """测试创建 ProcessInfo 实例"""
        info = ProcessInfo(
            pid=1234,
            name="test.exe",
            cmdline="test.exe --arg",
            create_time=datetime.now(),
            cpu_percent=5.5,
            memory_percent=2.0,
            runtime_minutes=45.0,
            is_long_running=True,
            process_type=ProcessType.USER_APP,
            protect_flag=True,
        )
        
        self.assertEqual(info.pid, 1234)
        self.assertEqual(info.name, "test.exe")
        self.assertTrue(info.is_long_running)
        self.assertTrue(info.protect_flag)
    
    def test_process_info_to_dict(self):
        """测试 ProcessInfo 转换为字典"""
        info = ProcessInfo(
            pid=1234,
            name="test.exe",
            cmdline="test.exe --arg",
            create_time=datetime(2024, 1, 1, 12, 0, 0),
            cpu_percent=5.5,
            memory_percent=2.0,
            runtime_minutes=45.0,
            is_long_running=True,
            process_type=ProcessType.TRAINING,
            protect_flag=True,
            username="testuser",
            memory_mb=100.5,
        )
        
        data = info.to_dict()
        
        self.assertEqual(data["pid"], 1234)
        self.assertEqual(data["name"], "test.exe")
        self.assertEqual(data["process_type"], "training")
        self.assertTrue(data["is_long_running"])
        self.assertEqual(data["username"], "testuser")
        self.assertEqual(data["memory_mb"], 100.5)


class TestProcessScanner(unittest.TestCase):
    """测试 ProcessScanner 类"""
    
    def setUp(self):
        """测试前准备"""
        reset_scanner()
        self.scanner = ProcessScanner(
            long_running_threshold=30,
            scan_interval_minutes=10,
        )
    
    def tearDown(self):
        """测试后清理"""
        reset_scanner()
    
    def test_scanner_initialization(self):
        """测试扫描器初始化"""
        self.assertEqual(self.scanner._long_running_threshold, 30)
        self.assertEqual(self.scanner._scan_interval_minutes, 10)
        self.assertIsNone(self.scanner._last_scan_time)
        self.assertEqual(len(self.scanner._process_cache), 0)
    
    def test_scanner_with_custom_whitelist(self):
        """测试带自定义白名单的扫描器"""
        custom_whitelist = {"custom_app.exe", "my_service.exe"}
        scanner = ProcessScanner(custom_whitelist=custom_whitelist)
        
        self.assertIn("custom_app.exe", scanner._all_whitelist)
        self.assertIn("my_service.exe", scanner._all_whitelist)
    
    @patch('psutil.process_iter')
    def test_scan_basic(self, mock_process_iter):
        """测试基本扫描功能"""
        # 模拟进程
        mock_proc = Mock()
        mock_proc.info = {
            'pid': 1234,
            'name': 'python.exe',
            'cmdline': ['python.exe', 'train.py'],
            'create_time': time.time() - 3600,  # 1小时前
            'cpu_percent': 10.0,
            'memory_percent': 5.0,
            'memory_info': Mock(rss=100*1024*1024),
            'username': 'testuser',
        }
        mock_process_iter.return_value = [mock_proc]
        
        # 执行扫描
        result = self.scanner.scan(force=True)
        
        # 验证结果
        self.assertIsInstance(result, list)
        mock_process_iter.assert_called_once()
    
    @patch('psutil.process_iter')
    def test_scan_respects_interval(self, mock_process_iter):
        """测试扫描间隔限制"""
        mock_process_iter.return_value = []
        
        # 第一次扫描
        self.scanner.scan(force=True)
        first_scan_time = self.scanner._last_scan_time
        
        # 立即再次扫描（应该跳过）
        result = self.scanner.scan(force=False)
        
        # 应该返回缓存结果，不执行新扫描
        self.assertEqual(self.scanner._last_scan_time, first_scan_time)
    
    def test_classify_process_training(self):
        """测试训练进程分类"""
        proc_type = self.scanner._classify_process(
            "python.exe", "python train.py"
        )
        self.assertEqual(proc_type, ProcessType.TRAINING)
    
    def test_classify_process_rendering(self):
        """测试渲染进程分类"""
        proc_type = self.scanner._classify_process(
            "blender.exe", "blender render.blend"
        )
        self.assertEqual(proc_type, ProcessType.RENDERING)
    
    def test_classify_process_system(self):
        """测试系统进程分类"""
        proc_type = self.scanner._classify_process(
            "svchost.exe", "svchost.exe"
        )
        self.assertEqual(proc_type, ProcessType.SYSTEM)
    
    def test_should_protect_training(self):
        """测试训练进程保护判断"""
        should_protect = self.scanner._should_protect(
            "python.exe", "python train.py", ProcessType.TRAINING, 1234
        )
        self.assertTrue(should_protect)
    
    def test_should_protect_system(self):
        """测试系统进程保护判断"""
        should_protect = self.scanner._should_protect(
            "svchost.exe", "svchost.exe", ProcessType.SYSTEM, 1234
        )
        self.assertTrue(should_protect)
    
    def test_mark_protected(self):
        """测试标记进程为保护状态"""
        with patch('psutil.pid_exists', return_value=True):
            result = self.scanner.mark_protected(1234, "Test reason")
            self.assertTrue(result)
            self.assertIn(1234, self.scanner._protected_pids)
            self.assertEqual(self.scanner._protected_pids[1234], "Test reason")
    
    def test_mark_protected_nonexistent_pid(self):
        """测试标记不存在的进程"""
        with patch('psutil.pid_exists', return_value=False):
            result = self.scanner.mark_protected(99999, "Test reason")
            self.assertFalse(result)
    
    def test_unmark_protected(self):
        """测试取消保护标记"""
        with patch('psutil.pid_exists', return_value=True):
            self.scanner.mark_protected(1234, "Test reason")
            result = self.scanner.unmark_protected(1234)
            self.assertTrue(result)
            self.assertNotIn(1234, self.scanner._protected_pids)
    
    @patch('psutil.process_iter')
    def test_should_prevent_shutdown(self, mock_process_iter):
        """测试关机阻止判断"""
        # 模拟长时间运行的受保护进程
        mock_proc = Mock()
        mock_proc.info = {
            'pid': 1234,
            'name': 'python.exe',
            'cmdline': ['python.exe', 'train.py'],
            'create_time': time.time() - 3600,
            'cpu_percent': 10.0,
            'memory_percent': 5.0,
            'memory_info': Mock(rss=100*1024*1024),
            'username': 'testuser',
        }
        mock_process_iter.return_value = [mock_proc]
        
        # 执行扫描
        self.scanner.scan(force=True)
        
        # 检查是否应该阻止关机
        should_block, reasons = self.scanner.should_prevent_shutdown()
        
        # 应该阻止关机（因为有训练任务）
        self.assertTrue(should_block)
        self.assertGreater(len(reasons), 0)
    
    def test_get_scan_stats(self):
        """测试获取扫描统计信息"""
        stats = self.scanner.get_scan_stats()
        
        self.assertIn("last_scan_time", stats)
        self.assertIn("total_processes", stats)
        self.assertIn("long_running_tasks", stats)
        self.assertIn("scan_interval_minutes", stats)
        self.assertEqual(stats["scan_interval_minutes"], 10)
    
    def test_add_to_whitelist(self):
        """测试添加进程到白名单"""
        self.scanner.add_to_whitelist("my_custom_app.exe")
        self.assertIn("my_custom_app.exe", self.scanner._custom_whitelist)
        self.assertIn("my_custom_app.exe", self.scanner._all_whitelist)
    
    def test_remove_from_whitelist(self):
        """测试从白名单移除进程"""
        self.scanner.add_to_whitelist("my_custom_app.exe")
        self.scanner.remove_from_whitelist("my_custom_app.exe")
        self.assertNotIn("my_custom_app.exe", self.scanner._custom_whitelist)


class TestWhitelistManager(unittest.TestCase):
    """测试 WhitelistManager 类"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.yaml")
    
    def tearDown(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_whitelist_manager_initialization(self):
        """测试白名单管理器初始化"""
        manager = WhitelistManager(config_path=self.config_path)
        
        self.assertGreater(len(manager.get_system_whitelist()), 0)
        self.assertEqual(len(manager.get_user_whitelist()), 0)
    
    def test_is_whitelisted(self):
        """测试白名单检查"""
        manager = WhitelistManager(config_path=self.config_path)
        
        # 系统进程应该在白名单中
        self.assertTrue(manager.is_whitelisted("svchost.exe"))
        self.assertTrue(manager.is_whitelisted("python.exe"))
        
        # 随机进程名不应该在白名单中
        self.assertFalse(manager.is_whitelisted("random_app.exe"))
    
    def test_add_user_process(self):
        """测试添加用户进程"""
        manager = WhitelistManager(config_path=self.config_path)
        
        result = manager.add_user_process("my_app.exe")
        self.assertTrue(result)
        self.assertTrue(manager.is_whitelisted("my_app.exe"))
    
    def test_remove_user_process(self):
        """测试移除用户进程"""
        manager = WhitelistManager(config_path=self.config_path)
        
        manager.add_user_process("my_app.exe")
        result = manager.remove_user_process("my_app.exe")
        self.assertTrue(result)
        self.assertFalse(manager.is_whitelisted("my_app.exe"))
    
    def test_add_pattern(self):
        """测试添加正则表达式模式"""
        manager = WhitelistManager(config_path=self.config_path)
        
        result = manager.add_pattern(r".*test.*")
        self.assertTrue(result)
        self.assertTrue(manager.is_whitelisted("my_test_app.exe"))
    
    def test_add_invalid_pattern(self):
        """测试添加无效的正则表达式"""
        manager = WhitelistManager(config_path=self.config_path)
        
        result = manager.add_pattern(r"[invalid(")
        self.assertFalse(result)
    
    def test_export_import(self):
        """测试导出和导入"""
        manager = WhitelistManager(config_path=self.config_path)
        manager.add_user_process("export_test.exe", save=False)
        
        # 导出
        export_path = os.path.join(self.temp_dir, "export.yaml")
        result = manager.export_to_file(export_path)
        self.assertTrue(result)
        self.assertTrue(os.path.exists(export_path))
        
        # 创建新管理器并导入
        manager2 = WhitelistManager(config_path=self.config_path + ".2")
        result = manager2.import_from_file(export_path)
        self.assertTrue(result)
        self.assertTrue(manager2.is_whitelisted("export_test.exe"))
    
    def test_get_stats(self):
        """测试获取统计信息"""
        manager = WhitelistManager(config_path=self.config_path)
        
        stats = manager.get_stats()
        self.assertIn("system_processes_count", stats)
        self.assertIn("user_processes_count", stats)
        self.assertIn("patterns_count", stats)


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def setUp(self):
        """测试前准备"""
        reset_scanner()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """测试后清理"""
        reset_scanner()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_scanner_from_config(self):
        """测试从配置创建扫描器"""
        # 创建测试配置文件
        config_path = os.path.join(self.temp_dir, "test_config.yaml")
        config_data = {
            'scanner': {
                'long_running_threshold': 45,
                'scan_interval_minutes': 15,
            }
        }
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        # 从配置创建扫描器
        scanner = create_scanner_from_config(config_path)
        
        self.assertEqual(scanner._long_running_threshold, 45)
        self.assertEqual(scanner._scan_interval_minutes, 15)
    
    def test_get_scanner_singleton(self):
        """测试扫描器单例模式"""
        scanner1 = get_scanner()
        scanner2 = get_scanner()
        
        self.assertIs(scanner1, scanner2)
    
    @patch('psutil.process_iter')
    def test_quick_check(self, mock_process_iter):
        """测试快速检查功能"""
        reset_scanner()
        
        # 模拟进程
        mock_proc = Mock()
        mock_proc.info = {
            'pid': 1234,
            'name': 'python.exe',
            'cmdline': ['python.exe', 'train.py'],
            'create_time': time.time() - 3600,
            'cpu_percent': 10.0,
            'memory_percent': 5.0,
            'memory_info': Mock(rss=100*1024*1024),
            'username': 'testuser',
        }
        mock_process_iter.return_value = [mock_proc]
        
        # 快速检查
        should_block, reasons = quick_check()
        
        self.assertIsInstance(should_block, bool)
        self.assertIsInstance(reasons, list)


class TestEdgeCases(unittest.TestCase):
    """边界情况测试"""
    
    def setUp(self):
        """测试前准备"""
        reset_scanner()
        self.scanner = ProcessScanner()
    
    def tearDown(self):
        """测试后清理"""
        reset_scanner()
    
    def test_empty_process_name(self):
        """测试空进程名"""
        result = self.scanner._classify_process("", "")
        self.assertEqual(result, ProcessType.USER_APP)
    
    def test_none_process_info(self):
        """测试处理None进程信息"""
        with patch('psutil.process_iter') as mock_iter:
            mock_proc = Mock()
            mock_proc.info = None
            mock_iter.return_value = [mock_proc]
            
            # 不应该抛出异常
            result = self.scanner.scan(force=True)
            self.assertIsInstance(result, list)
    
    def test_access_denied_process(self):
        """测试访问被拒绝的进程"""
        with patch('psutil.process_iter') as mock_iter:
            mock_proc = Mock()
            mock_proc.info = {'pid': 1234}
            mock_iter.return_value = [mock_proc]
            
            with patch.object(self.scanner, '_analyze_process', 
                            side_effect=psutil.AccessDenied):
                # 不应该抛出异常
                result = self.scanner.scan(force=True)
                self.assertIsInstance(result, list)
    
    def test_no_such_process(self):
        """测试不存在的进程"""
        with patch('psutil.process_iter') as mock_iter:
            mock_proc = Mock()
            mock_proc.info = {'pid': 1234}
            mock_iter.return_value = [mock_proc]
            
            with patch.object(self.scanner, '_analyze_process', 
                            side_effect=psutil.NoSuchProcess(1234)):
                # 不应该抛出异常
                result = self.scanner.scan(force=True)
                self.assertIsInstance(result, list)


def run_tests():
    """运行所有测试"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加所有测试类
    suite.addTests(loader.loadTestsFromTestCase(TestProcessInfo))
    suite.addTests(loader.loadTestsFromTestCase(TestProcessScanner))
    suite.addTests(loader.loadTestsFromTestCase(TestWhitelistManager))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
