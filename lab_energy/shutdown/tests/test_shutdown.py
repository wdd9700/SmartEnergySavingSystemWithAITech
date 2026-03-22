"""
智能关机决策系统单元测试
"""

import unittest
from datetime import datetime, time
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lab_energy.shutdown import (
    ShutdownDecision,
    RiskLevel,
    ShutdownContext,
    DecisionResult,
    ShutdownDecisionEngine,
    NotificationConfig,
    ConsoleNotifier,
    ShutdownExecutor,
    ScheduledShutdownExecutor,
)


class TestShutdownDecision(unittest.TestCase):
    """测试关机决策枚举"""

    def test_shutdown_decision_values(self):
        """测试决策枚举值"""
        self.assertEqual(ShutdownDecision.SHUTDOWN.value, "shutdown")
        self.assertEqual(ShutdownDecision.DELAY.value, "delay")
        self.assertEqual(ShutdownDecision.CANCEL.value, "cancel")
        self.assertEqual(ShutdownDecision.NOTIFY.value, "notify")


class TestRiskLevel(unittest.TestCase):
    """测试风险等级枚举"""

    def test_risk_level_values(self):
        """测试风险等级值"""
        self.assertEqual(RiskLevel.LOW.value, 1)
        self.assertEqual(RiskLevel.MEDIUM.value, 2)
        self.assertEqual(RiskLevel.HIGH.value, 3)


class TestShutdownContext(unittest.TestCase):
    """测试关机上下文数据类"""

    def test_context_creation(self):
        """测试上下文创建"""
        now = datetime.now()
        context = ShutdownContext(
            timestamp=now,
            long_running_tasks=[{"name": "test", "pid": 1234}],
            user_active=True,
            time_of_day=now.time(),
            day_of_week=0,
            gpu_utilization=30.0,
            cpu_utilization=50.0
        )

        self.assertEqual(context.timestamp, now)
        self.assertEqual(len(context.long_running_tasks), 1)
        self.assertTrue(context.user_active)
        self.assertEqual(context.day_of_week, 0)
        self.assertEqual(context.gpu_utilization, 30.0)
        self.assertEqual(context.cpu_utilization, 50.0)


class TestDecisionResult(unittest.TestCase):
    """测试决策结果数据类"""

    def test_result_creation(self):
        """测试结果创建"""
        result = DecisionResult(
            decision=ShutdownDecision.SHUTDOWN,
            risk_level=RiskLevel.LOW,
            reasons=["test reason"],
            countdown_minutes=0,
            can_cancel=False,
            risk_score=1
        )

        self.assertEqual(result.decision, ShutdownDecision.SHUTDOWN)
        self.assertEqual(result.risk_level, RiskLevel.LOW)
        self.assertEqual(result.reasons, ["test reason"])
        self.assertEqual(result.countdown_minutes, 0)
        self.assertFalse(result.can_cancel)
        self.assertEqual(result.risk_score, 1)


class TestShutdownDecisionEngine(unittest.TestCase):
    """测试关机决策引擎"""

    def setUp(self):
        """测试前准备"""
        self.engine = ShutdownDecisionEngine()

    def test_initialization(self):
        """测试引擎初始化"""
        self.assertIsNone(self.engine._scanner)
        self.assertEqual(len(self.engine._decision_history), 0)
        self.assertIsNone(self.engine._last_evaluation_time)

    def test_initialization_with_scanner(self):
        """测试带扫描器的初始化"""
        mock_scanner = Mock()
        engine = ShutdownDecisionEngine(mock_scanner)
        self.assertEqual(engine._scanner, mock_scanner)

    def test_is_work_hours_weekday(self):
        """测试工作日工作时间判断"""
        # 周一上午10点
        test_time = datetime(2024, 1, 1, 10, 0, 0)  # 周一
        self.assertTrue(self.engine.is_work_hours(test_time))

    def test_is_work_hours_weekend(self):
        """测试周末工作时间判断"""
        # 周六上午10点
        test_time = datetime(2024, 1, 6, 10, 0, 0)  # 周六
        self.assertFalse(self.engine.is_work_hours(test_time))

    def test_is_work_hours_before_work(self):
        """测试工作日前工作时间判断"""
        # 周一上午8点
        test_time = datetime(2024, 1, 1, 8, 0, 0)  # 周一
        self.assertFalse(self.engine.is_work_hours(test_time))

    def test_is_work_hours_after_work(self):
        """测试工作日后工作时间判断"""
        # 周一晚上8点
        test_time = datetime(2024, 1, 1, 20, 0, 0)  # 周一
        self.assertFalse(self.engine.is_work_hours(test_time))

    def test_calculate_risk_score_no_risk(self):
        """测试无风险评分"""
        now = datetime.now()
        context = ShutdownContext(
            timestamp=now,
            long_running_tasks=[],
            user_active=False,
            time_of_day=time(20, 0),  # 晚上8点
            day_of_week=5,  # 周六
            gpu_utilization=30.0,
            cpu_utilization=50.0
        )

        score = self.engine.calculate_risk_score(context)
        self.assertEqual(score, 0)

    def test_calculate_risk_score_with_long_tasks(self):
        """测试有长时间任务的评分"""
        now = datetime.now()
        context = ShutdownContext(
            timestamp=now,
            long_running_tasks=[{"name": "task1", "pid": 1234}],
            user_active=False,
            time_of_day=time(20, 0),
            day_of_week=5,
            gpu_utilization=30.0,
            cpu_utilization=50.0
        )

        score = self.engine.calculate_risk_score(context)
        self.assertEqual(score, 3)  # 长时间任务 +3

    def test_calculate_risk_score_user_active(self):
        """测试用户活跃的评分"""
        now = datetime.now()
        context = ShutdownContext(
            timestamp=now,
            long_running_tasks=[],
            user_active=True,
            time_of_day=time(20, 0),
            day_of_week=5,
            gpu_utilization=30.0,
            cpu_utilization=50.0
        )

        score = self.engine.calculate_risk_score(context)
        self.assertEqual(score, 2)  # 用户活跃 +2

    def test_calculate_risk_score_work_hours(self):
        """测试工作时间的评分"""
        now = datetime.now()
        context = ShutdownContext(
            timestamp=now,
            long_running_tasks=[],
            user_active=False,
            time_of_day=time(10, 0),  # 上午10点
            day_of_week=0,  # 周一
            gpu_utilization=30.0,
            cpu_utilization=50.0
        )

        score = self.engine.calculate_risk_score(context)
        self.assertEqual(score, 1)  # 工作时间 +1

    def test_calculate_risk_score_high_gpu(self):
        """测试高GPU利用率的评分"""
        now = datetime.now()
        context = ShutdownContext(
            timestamp=now,
            long_running_tasks=[],
            user_active=False,
            time_of_day=time(20, 0),
            day_of_week=5,
            gpu_utilization=60.0,  # >50%
            cpu_utilization=50.0
        )

        score = self.engine.calculate_risk_score(context)
        self.assertEqual(score, 2)  # GPU高 +2

    def test_calculate_risk_score_high_cpu(self):
        """测试高CPU利用率的评分"""
        now = datetime.now()
        context = ShutdownContext(
            timestamp=now,
            long_running_tasks=[],
            user_active=False,
            time_of_day=time(20, 0),
            day_of_week=5,
            gpu_utilization=30.0,
            cpu_utilization=80.0  # >70%
        )

        score = self.engine.calculate_risk_score(context)
        self.assertEqual(score, 1)  # CPU高 +1

    def test_calculate_risk_score_multiple_factors(self):
        """测试多因素评分"""
        now = datetime.now()
        context = ShutdownContext(
            timestamp=now,
            long_running_tasks=[{"name": "task1"}],
            user_active=True,
            time_of_day=time(10, 0),
            day_of_week=0,
            gpu_utilization=60.0,
            cpu_utilization=80.0
        )

        score = self.engine.calculate_risk_score(context)
        # 长时间任务(3) + 用户活跃(2) + 工作时间(1) + GPU高(2) + CPU高(1) = 9
        self.assertEqual(score, 9)

    def test_evaluate_shutdown(self):
        """测试评估关机决策"""
        now = datetime.now()
        context = ShutdownContext(
            timestamp=now,
            long_running_tasks=[],
            user_active=False,
            time_of_day=time(20, 0),
            day_of_week=5,
            gpu_utilization=30.0,
            cpu_utilization=50.0
        )

        result = self.engine.evaluate(context)

        self.assertEqual(result.decision, ShutdownDecision.SHUTDOWN)
        self.assertEqual(result.risk_level, RiskLevel.LOW)
        self.assertEqual(result.countdown_minutes, 0)
        self.assertFalse(result.can_cancel)

    def test_evaluate_notify_short(self):
        """测试评估通知（短倒计时）"""
        now = datetime.now()
        context = ShutdownContext(
            timestamp=now,
            long_running_tasks=[],
            user_active=True,  # +2
            time_of_day=time(20, 0),
            day_of_week=5,
            gpu_utilization=30.0,
            cpu_utilization=50.0
        )

        result = self.engine.evaluate(context)

        self.assertEqual(result.decision, ShutdownDecision.NOTIFY)
        self.assertEqual(result.risk_level, RiskLevel.MEDIUM)
        self.assertEqual(result.countdown_minutes, 10)
        self.assertTrue(result.can_cancel)

    def test_evaluate_notify_long(self):
        """测试评估通知（长倒计时）"""
        now = datetime.now()
        context = ShutdownContext(
            timestamp=now,
            long_running_tasks=[{"name": "task1"}],  # +3
            user_active=True,  # +2
            time_of_day=time(20, 0),
            day_of_week=5,
            gpu_utilization=30.0,
            cpu_utilization=50.0
        )

        result = self.engine.evaluate(context)

        self.assertEqual(result.decision, ShutdownDecision.NOTIFY)
        self.assertEqual(result.risk_level, RiskLevel.MEDIUM)
        self.assertEqual(result.countdown_minutes, 30)
        self.assertTrue(result.can_cancel)

    def test_evaluate_cancel(self):
        """测试评估取消（高风险）"""
        now = datetime.now()
        context = ShutdownContext(
            timestamp=now,
            long_running_tasks=[{"name": "task1"}],  # +3
            user_active=True,  # +2
            time_of_day=time(10, 0),  # 工作时间 +1
            day_of_week=0,
            gpu_utilization=60.0,  # +2
            cpu_utilization=80.0  # +1
        )

        result = self.engine.evaluate(context)

        self.assertEqual(result.decision, ShutdownDecision.CANCEL)
        self.assertEqual(result.risk_level, RiskLevel.HIGH)
        self.assertEqual(result.countdown_minutes, 0)
        self.assertFalse(result.can_cancel)

    def test_should_shutdown(self):
        """测试简化关机判断"""
        now = datetime.now()
        context = ShutdownContext(
            timestamp=now,
            long_running_tasks=[],
            user_active=False,
            time_of_day=time(20, 0),
            day_of_week=5,
            gpu_utilization=30.0,
            cpu_utilization=50.0
        )

        should_shutdown, reasons = self.engine.should_shutdown()
        # 注意：should_shutdown 会调用 evaluate() 而不传参数，结果取决于当前实际状态
        # 这里我们只测试返回值类型
        self.assertIsInstance(should_shutdown, bool)
        self.assertIsInstance(reasons, list)

    def test_decision_history(self):
        """测试决策历史记录"""
        now = datetime.now()
        context = ShutdownContext(
            timestamp=now,
            long_running_tasks=[],
            user_active=False,
            time_of_day=time(20, 0),
            day_of_week=5,
            gpu_utilization=30.0,
            cpu_utilization=50.0
        )

        # 执行多次评估
        self.engine.evaluate(context)
        self.engine.evaluate(context)
        self.engine.evaluate(context)

        history = self.engine.get_decision_history()
        self.assertEqual(len(history), 3)

    def test_clear_history(self):
        """测试清空历史记录"""
        now = datetime.now()
        context = ShutdownContext(
            timestamp=now,
            long_running_tasks=[],
            user_active=False,
            time_of_day=time(20, 0),
            day_of_week=5,
            gpu_utilization=30.0,
            cpu_utilization=50.0
        )

        self.engine.evaluate(context)
        self.engine.clear_history()

        history = self.engine.get_decision_history()
        self.assertEqual(len(history), 0)

    def test_update_work_hours(self):
        """测试更新工作时间"""
        new_start = time(8, 30)
        new_end = time(17, 30)
        new_workdays = {0, 1, 2, 3, 4, 5}  # 周一到周六

        self.engine.update_work_hours(new_start, new_end, new_workdays)

        self.assertEqual(self.engine.WORK_HOURS['start'], new_start)
        self.assertEqual(self.engine.WORK_HOURS['end'], new_end)
        self.assertEqual(self.engine.WORK_HOURS['workdays'], new_workdays)


class TestNotificationConfig(unittest.TestCase):
    """测试通知配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = NotificationConfig()
        self.assertEqual(config.title, "智能关机提醒")
        self.assertEqual(config.app_name, "Lab Energy Manager")
        self.assertIsNone(config.icon_path)
        self.assertEqual(config.duration, 10)

    def test_custom_config(self):
        """测试自定义配置"""
        config = NotificationConfig(
            title="Test Title",
            app_name="Test App",
            icon_path="test.ico",
            duration=20
        )
        self.assertEqual(config.title, "Test Title")
        self.assertEqual(config.app_name, "Test App")
        self.assertEqual(config.icon_path, "test.ico")
        self.assertEqual(config.duration, 20)


class TestConsoleNotifier(unittest.TestCase):
    """测试控制台通知器"""

    def setUp(self):
        """测试前准备"""
        self.notifier = ConsoleNotifier()

    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.notifier._config)
        self.assertFalse(self.notifier._cancelled)
        self.assertFalse(self.notifier._countdown_active)

    def test_notify_shutdown_pending(self):
        """测试关机通知"""
        result = self.notifier.notify_shutdown_pending(
            countdown_minutes=10,
            reasons=["test reason"]
        )
        self.assertTrue(result)

    def test_notify_shutdown_cancelled(self):
        """测试取消通知"""
        result = self.notifier.notify_shutdown_cancelled("test cancel")
        self.assertTrue(result)

    def test_notify_shutdown_executed(self):
        """测试执行通知"""
        result = self.notifier.notify_shutdown_executed()
        self.assertTrue(result)

    def test_cancel_countdown(self):
        """测试取消倒计时"""
        # 先启动一个倒计时
        self.notifier._countdown_active = True
        result = self.notifier.cancel_countdown()
        self.assertTrue(result)
        self.assertTrue(self.notifier._cancelled)

    def test_is_countdown_active(self):
        """测试检查倒计时状态"""
        self.assertFalse(self.notifier.is_countdown_active())
        self.notifier._countdown_active = True
        self.assertTrue(self.notifier.is_countdown_active())


class TestShutdownExecutor(unittest.TestCase):
    """测试关机执行器"""

    def setUp(self):
        """测试前准备"""
        self.mock_engine = Mock(spec=ShutdownDecisionEngine)
        self.mock_notifier = Mock(spec=ConsoleNotifier)
        self.executor = ShutdownExecutor(
            decision_engine=self.mock_engine,
            notifier=self.mock_notifier,
            use_console_notifier=True
        )

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.executor._engine, self.mock_engine)
        self.assertEqual(self.executor._notifier, self.mock_notifier)
        self.assertFalse(self.executor._cancelled)
        self.assertFalse(self.executor._shutdown_in_progress)

    def test_is_shutdown_in_progress(self):
        """测试检查关机状态"""
        self.assertFalse(self.executor.is_shutdown_in_progress())
        self.executor._shutdown_in_progress = True
        self.assertTrue(self.executor.is_shutdown_in_progress())

    def test_is_cancelled(self):
        """测试检查取消状态"""
        self.assertFalse(self.executor.is_cancelled())
        self.executor._cancelled = True
        self.assertTrue(self.executor.is_cancelled())

    def test_add_shutdown_callback(self):
        """测试添加关机回调"""
        callback = Mock()
        self.executor.add_shutdown_callback(callback)
        self.assertIn(callback, self.executor._on_shutdown_callbacks)

    def test_add_cancel_callback(self):
        """测试添加取消回调"""
        callback = Mock()
        self.executor.add_cancel_callback(callback)
        self.assertIn(callback, self.executor._on_cancel_callbacks)

    def test_remove_shutdown_callback(self):
        """测试移除关机回调"""
        callback = Mock()
        self.executor.add_shutdown_callback(callback)
        self.executor.remove_shutdown_callback(callback)
        self.assertNotIn(callback, self.executor._on_shutdown_callbacks)

    def test_remove_cancel_callback(self):
        """测试移除取消回调"""
        callback = Mock()
        self.executor.add_cancel_callback(callback)
        self.executor.remove_cancel_callback(callback)
        self.assertNotIn(callback, self.executor._on_cancel_callbacks)

    @patch('subprocess.run')
    def test_abort_shutdown(self, mock_run):
        """测试中止关机"""
        mock_run.return_value = Mock(returncode=0)
        result = self.executor.abort_shutdown()
        self.assertTrue(result)
        mock_run.assert_called_once()


class TestScheduledShutdownExecutor(unittest.TestCase):
    """测试定时关机执行器"""

    def setUp(self):
        """测试前准备"""
        self.mock_engine = Mock(spec=ShutdownDecisionEngine)
        self.executor = ScheduledShutdownExecutor(
            decision_engine=self.mock_engine,
            evaluation_interval=1  # 1秒间隔用于测试
        )

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.executor._engine, self.mock_engine)
        self.assertEqual(self.executor._evaluation_interval, 1)
        self.assertFalse(self.executor._running)

    def test_is_running(self):
        """测试检查运行状态"""
        self.assertFalse(self.executor.is_running())
        self.executor._running = True
        self.assertTrue(self.executor.is_running())

    def test_update_interval(self):
        """测试更新间隔"""
        self.executor.update_interval(300)
        self.assertEqual(self.executor._evaluation_interval, 300)


class TestIntegration(unittest.TestCase):
    """集成测试"""

    def test_full_workflow_low_risk(self):
        """测试完整工作流 - 低风险"""
        engine = ShutdownDecisionEngine()
        executor = ShutdownExecutor(engine, use_console_notifier=True)

        # 模拟低风险上下文（周末晚上）
        now = datetime.now()
        context = ShutdownContext(
            timestamp=now,
            long_running_tasks=[],
            user_active=False,
            time_of_day=time(20, 0),
            day_of_week=5,
            gpu_utilization=30.0,
            cpu_utilization=50.0
        )

        result = engine.evaluate(context)
        self.assertEqual(result.decision, ShutdownDecision.SHUTDOWN)
        self.assertEqual(result.risk_level, RiskLevel.LOW)

    def test_full_workflow_high_risk(self):
        """测试完整工作流 - 高风险"""
        engine = ShutdownDecisionEngine()
        executor = ShutdownExecutor(engine, use_console_notifier=True)

        # 模拟高风险上下文（工作时间，用户活跃，有长时间任务）
        now = datetime.now()
        context = ShutdownContext(
            timestamp=now,
            long_running_tasks=[{"name": "training", "pid": 1234}],
            user_active=True,
            time_of_day=time(10, 0),
            day_of_week=0,
            gpu_utilization=80.0,
            cpu_utilization=90.0
        )

        result = engine.evaluate(context)
        self.assertEqual(result.decision, ShutdownDecision.CANCEL)
        self.assertEqual(result.risk_level, RiskLevel.HIGH)


if __name__ == "__main__":
    # 设置日志级别
    logging.basicConfig(level=logging.WARNING)

    # 运行测试
    unittest.main(verbosity=2)
