"""
关机执行器

负责:
1. 用户通知
2. 倒计时
3. 关机前保存
4. 执行关机
"""

import logging
import os
import subprocess
import threading
import time
from typing import Callable, List, Optional, Tuple

from .decision_engine import ShutdownDecisionEngine, ShutdownDecision, DecisionResult
from .user_notifier import UserNotifier, ConsoleNotifier, NotificationConfig

logger = logging.getLogger(__name__)


class ShutdownExecutor:
    """
    关机执行器

    负责:
    1. 用户通知
    2. 倒计时
    3. 关机前保存
    4. 执行关机
    """

    # 关机前保存工作的超时时间（秒）
    SAVE_WORK_TIMEOUT = 30

    def __init__(
        self,
        decision_engine: ShutdownDecisionEngine,
        notifier: Optional[UserNotifier] = None,
        use_console_notifier: bool = False
    ):
        """
        Args:
            decision_engine: 决策引擎实例
            notifier: 用户通知器（可选）
            use_console_notifier: 是否使用控制台通知器
        """
        self._engine = decision_engine
        self._notifier = notifier or (
            ConsoleNotifier() if use_console_notifier else UserNotifier()
        )
        self._cancelled = False
        self._countdown_thread: Optional[threading.Thread] = None
        self._shutdown_in_progress = False
        self._on_shutdown_callbacks: List[Callable[[], None]] = []
        self._on_cancel_callbacks: List[Callable[[], None]] = []

    def notify_user(self, message: str, timeout_minutes: int) -> bool:
        """
        通知用户即将关机

        使用Windows Toast通知

        Args:
            message: 通知消息
            timeout_minutes: 倒计时分钟

        Returns:
            通知是否成功显示
        """
        try:
            result = self._notifier.notify_shutdown_pending(
                countdown_minutes=timeout_minutes,
                reasons=[message],
                callback=lambda remaining: logger.debug(f"倒计时: {remaining}分钟")
            )
            return result
        except Exception as e:
            logger.error(f"通知用户失败: {e}")
            return False

    def countdown(
        self,
        minutes: int,
        callback: Optional[Callable[[int], None]] = None
    ) -> bool:
        """
        倒计时等待

        Args:
            minutes: 倒计时分钟
            callback: 每分钟回调函数，参数为剩余分钟数

        Returns:
            是否正常完成（False=用户取消）
        """
        self._cancelled = False
        self._shutdown_in_progress = True

        def tick_callback(remaining: int):
            logger.info(f"关机倒计时: {remaining}分钟")
            if callback:
                try:
                    callback(remaining)
                except Exception as e:
                    logger.warning(f"倒计时回调失败: {e}")

        def finished_callback():
            logger.info("倒计时结束")

        self._countdown_thread = self._notifier.start_countdown(
            minutes=minutes,
            tick_callback=tick_callback,
            finished_callback=finished_callback
        )

        # 等待倒计时完成或被取消
        self._countdown_thread.join()

        self._shutdown_in_progress = False

        if self._cancelled:
            logger.info("倒计时被取消")
            return False

        return True

    def cancel_shutdown(self):
        """取消关机"""
        if not self._shutdown_in_progress:
            logger.warning("没有正在进行的关机流程")
            return

        self._cancelled = True
        self._notifier.cancel_countdown()

        # 执行取消回调
        for callback in self._on_cancel_callbacks:
            try:
                callback()
            except Exception as e:
                logger.warning(f"取消回调执行失败: {e}")

        logger.info("关机已取消")

    def save_user_work(self) -> bool:
        """
        尝试保存用户工作

        - 发送WM_SAVE消息到所有应用
        - 触发系统休眠前的保存流程

        Returns:
            是否成功
        """
        try:
            logger.info("正在尝试保存用户工作...")

            # 方法1: 尝试发送WM_SAVE消息到所有顶层窗口
            self._send_save_message_to_windows()

            # 方法2: 尝试最小化所有窗口（触发一些应用的自动保存）
            self._minimize_all_windows()

            # 等待一段时间让应用保存
            time.sleep(2)

            logger.info("保存用户工作完成")
            return True

        except Exception as e:
            logger.error(f"保存用户工作失败: {e}")
            return False

    def _send_save_message_to_windows(self):
        """发送保存消息到所有窗口"""
        try:
            import ctypes
            from ctypes import wintypes

            # 定义Windows API常量
            WM_SAVE = 0x0111  # 命令消息
            SC_SAVE = 0xF140  # 保存命令

            # 枚举所有顶层窗口
            EnumWindows = ctypes.windll.user32.EnumWindows
            EnumWindowsProc = ctypes.WINFUNCTYPE(
                ctypes.c_bool,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int)
            )

            def foreach_window(hwnd, lParam):
                # 发送保存消息
                ctypes.windll.user32.SendMessageW(hwnd, WM_SAVE, SC_SAVE, 0)
                return True

            EnumWindows(EnumWindowsProc(foreach_window), 0)

        except Exception as e:
            logger.warning(f"发送保存消息失败: {e}")

    def _minimize_all_windows(self):
        """最小化所有窗口"""
        try:
            import ctypes
            # 使用ShellExecute最小化所有窗口
            ctypes.windll.shell32.ShellExecuteW(
                None, "open", "explorer.exe", "/n,/select,::{20D04FE0-3AEA-1069-A2D8-08002B30309D}",
                None, 0
            )
        except Exception as e:
            logger.warning(f"最小化窗口失败: {e}")

    def execute_shutdown(self, force: bool = False) -> bool:
        """
        执行系统关机

        Args:
            force: 是否强制关机

        Returns:
            是否成功发起关机
        """
        try:
            logger.info(f"正在执行系统关机 (强制={force})...")

            # 通知用户即将关机
            self._notifier.notify_shutdown_executed()

            # 执行关机回调
            for callback in self._on_shutdown_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.warning(f"关机回调执行失败: {e}")

            # 构建关机命令
            if force:
                # 强制关机
                cmd = ["shutdown", "/s", "/f", "/t", "0"]
            else:
                # 正常关机，给用户30秒保存工作
                cmd = ["shutdown", "/s", "/t", "30", "/c", "系统即将关机，请保存您的工作"]

            # 执行关机命令
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                shell=True
            )

            if result.returncode == 0:
                logger.info("关机命令执行成功")
                return True
            else:
                logger.error(f"关机命令执行失败: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"执行关机失败: {e}")
            return False

    def run_shutdown_flow(self) -> bool:
        """
        执行完整的关机流程

        1. 评估决策
        2. 通知用户
        3. 倒计时
        4. 保存工作
        5. 执行关机

        Returns:
            是否成功执行关机
        """
        try:
            # 1. 评估决策
            result = self._engine.evaluate()
            logger.info(f"关机决策: {result.decision.value}, "
                       f"风险等级: {result.risk_level.name}")

            # 根据决策结果执行相应操作
            if result.decision == ShutdownDecision.SHUTDOWN:
                # 直接关机
                self.save_user_work()
                return self.execute_shutdown(force=False)

            elif result.decision == ShutdownDecision.NOTIFY:
                # 通知用户并倒计时
                self.notify_user(
                    message="; ".join(result.reasons),
                    timeout_minutes=result.countdown_minutes
                )

                # 倒计时
                countdown_success = self.countdown(result.countdown_minutes)

                if not countdown_success:
                    # 用户取消
                    logger.info("用户取消了关机")
                    return False

                # 倒计时结束，执行关机
                self.save_user_work()
                return self.execute_shutdown(force=False)

            elif result.decision == ShutdownDecision.DELAY:
                # 延迟检查
                logger.info("延迟关机检查")
                return False

            elif result.decision == ShutdownDecision.CANCEL:
                # 禁止关机
                logger.info("高风险状态，禁止关机")
                self._notifier.notify_shutdown_cancelled("检测到高风险任务正在运行")
                return False

            return False

        except Exception as e:
            logger.error(f"执行关机流程失败: {e}")
            return False

    def add_shutdown_callback(self, callback: Callable[[], None]):
        """
        添加关机回调函数

        Args:
            callback: 关机前执行的回调函数
        """
        self._on_shutdown_callbacks.append(callback)

    def add_cancel_callback(self, callback: Callable[[], None]):
        """
        添加取消回调函数

        Args:
            callback: 取消关机时执行的回调函数
        """
        self._on_cancel_callbacks.append(callback)

    def remove_shutdown_callback(self, callback: Callable[[], None]):
        """
        移除关机回调函数

        Args:
            callback: 要移除的回调函数
        """
        if callback in self._on_shutdown_callbacks:
            self._on_shutdown_callbacks.remove(callback)

    def remove_cancel_callback(self, callback: Callable[[], None]):
        """
        移除取消回调函数

        Args:
            callback: 要移除的回调函数
        """
        if callback in self._on_cancel_callbacks:
            self._on_cancel_callbacks.remove(callback)

    def is_shutdown_in_progress(self) -> bool:
        """
        检查是否正在执行关机流程

        Returns:
            是否正在关机
        """
        return self._shutdown_in_progress

    def is_cancelled(self) -> bool:
        """
        检查是否已取消

        Returns:
            是否已取消
        """
        return self._cancelled

    def abort_shutdown(self) -> bool:
        """
        中止已发起的关机命令

        Returns:
            是否成功中止
        """
        try:
            result = subprocess.run(
                ["shutdown", "/a"],
                capture_output=True,
                text=True,
                shell=True
            )

            if result.returncode == 0:
                logger.info("关机命令已中止")
                return True
            else:
                logger.warning(f"中止关机命令失败: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"中止关机失败: {e}")
            return False


class ScheduledShutdownExecutor(ShutdownExecutor):
    """
    定时关机执行器

    支持定时评估和自动关机
    """

    # 默认评估间隔（秒）
    DEFAULT_EVALUATION_INTERVAL = 600  # 10分钟

    def __init__(
        self,
        decision_engine: ShutdownDecisionEngine,
        notifier: Optional[UserNotifier] = None,
        evaluation_interval: int = DEFAULT_EVALUATION_INTERVAL,
        use_console_notifier: bool = False
    ):
        """
        Args:
            decision_engine: 决策引擎实例
            notifier: 用户通知器（可选）
            evaluation_interval: 评估间隔（秒）
            use_console_notifier: 是否使用控制台通知器
        """
        super().__init__(decision_engine, notifier, use_console_notifier)
        self._evaluation_interval = evaluation_interval
        self._scheduler_thread: Optional[threading.Thread] = None
        self._running = False

    def start_scheduler(self):
        """启动定时评估调度器"""
        if self._running:
            logger.warning("调度器已在运行")
            return

        self._running = True

        def scheduler_loop():
            while self._running:
                try:
                    logger.debug("执行定时关机评估...")

                    # 评估决策
                    result = self._engine.evaluate()

                    # 如果需要关机，执行关机流程
                    if result.decision == ShutdownDecision.SHUTDOWN:
                        logger.info("定时评估决定执行关机")
                        self.run_shutdown_flow()
                    elif result.decision == ShutdownDecision.NOTIFY:
                        logger.info(f"定时评估决定通知用户，倒计时: {result.countdown_minutes}分钟")
                        self.run_shutdown_flow()

                except Exception as e:
                    logger.error(f"定时评估失败: {e}")

                # 等待下一次评估
                for _ in range(self._evaluation_interval):
                    if not self._running:
                        break
                    time.sleep(1)

        self._scheduler_thread = threading.Thread(
            target=scheduler_loop,
            daemon=True
        )
        self._scheduler_thread.start()

        logger.info(f"定时关机调度器已启动，评估间隔: {self._evaluation_interval}秒")

    def stop_scheduler(self):
        """停止定时评估调度器"""
        self._running = False

        if self._scheduler_thread and self._scheduler_thread.is_alive():
            self._scheduler_thread.join(timeout=5)

        logger.info("定时关机调度器已停止")

    def is_running(self) -> bool:
        """
        检查调度器是否正在运行

        Returns:
            是否正在运行
        """
        return self._running

    def update_interval(self, interval: int):
        """
        更新评估间隔

        Args:
            interval: 新的评估间隔（秒）
        """
        self._evaluation_interval = interval
        logger.info(f"评估间隔已更新为: {interval}秒")
