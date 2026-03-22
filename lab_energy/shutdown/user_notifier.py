"""
用户通知模块

提供Windows Toast通知功能，用于通知用户即将关机。
"""

import logging
import threading
import time
from typing import Callable, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class NotificationType(Enum):
    """通知类型"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class NotificationConfig:
    """通知配置"""
    title: str = "智能关机提醒"
    app_name: str = "Lab Energy Manager"
    icon_path: Optional[str] = None
    duration: int = 10  # 通知显示时长（秒）


class UserNotifier:
    """
    用户通知器

    负责显示Windows Toast通知，提醒用户即将关机。
    支持倒计时通知和取消机制。
    """

    def __init__(self, config: Optional[NotificationConfig] = None):
        """
        Args:
            config: 通知配置
        """
        self._config = config or NotificationConfig()
        self._toast = None
        self._notification_thread: Optional[threading.Thread] = None
        self._cancelled = False
        self._countdown_active = False
        self._last_notification_time: Optional[float] = None

        # 尝试导入win10toast
        self._init_toast()

    def _init_toast(self):
        """初始化Toast通知器"""
        try:
            from win10toast import ToastNotifier
            self._toast = ToastNotifier()
            logger.info("Toast通知器初始化成功")
        except ImportError:
            logger.warning("win10toast未安装，将使用备用通知方式")
            self._toast = None
        except Exception as e:
            logger.warning(f"Toast通知器初始化失败: {e}")
            self._toast = None

    def notify_shutdown_pending(
        self,
        countdown_minutes: int,
        reasons: list,
        callback: Optional[Callable[[int], None]] = None
    ) -> bool:
        """
        通知用户即将关机

        Args:
            countdown_minutes: 倒计时分钟数
            reasons: 关机原因列表
            callback: 每分钟回调函数，参数为剩余分钟数

        Returns:
            通知是否成功显示
        """
        try:
            reason_text = "; ".join(reasons[:3]) if reasons else "系统空闲"

            message = (
                f"系统将在 {countdown_minutes} 分钟后关机\n"
                f"原因: {reason_text}\n"
                f"点击取消以阻止关机"
            )

            success = self._show_notification(
                title=self._config.title,
                message=message,
                icon_path=self._config.icon_path,
                duration=self._config.duration
            )

            if success:
                self._last_notification_time = time.time()
                logger.info(f"已发送关机通知: {countdown_minutes}分钟后关机")

            return success

        except Exception as e:
            logger.error(f"发送关机通知失败: {e}")
            return False

    def notify_shutdown_cancelled(self, reason: str = "用户取消") -> bool:
        """
        通知用户关机已取消

        Args:
            reason: 取消原因

        Returns:
            通知是否成功显示
        """
        try:
            message = f"关机已取消\n原因: {reason}"

            success = self._show_notification(
                title=self._config.title,
                message=message,
                icon_path=self._config.icon_path,
                duration=5
            )

            if success:
                logger.info(f"已发送关机取消通知: {reason}")

            return success

        except Exception as e:
            logger.error(f"发送取消通知失败: {e}")
            return False

    def notify_shutdown_executed(self) -> bool:
        """
        通知用户系统即将关机

        Returns:
            通知是否成功显示
        """
        try:
            message = "系统正在关机，请保存您的工作"

            success = self._show_notification(
                title=self._config.title,
                message=message,
                icon_path=self._config.icon_path,
                duration=10
            )

            if success:
                logger.info("已发送关机执行通知")

            return success

        except Exception as e:
            logger.error(f"发送关机执行通知失败: {e}")
            return False

    def start_countdown(
        self,
        minutes: int,
        tick_callback: Optional[Callable[[int], None]] = None,
        finished_callback: Optional[Callable[[], None]] = None
    ) -> threading.Thread:
        """
        开始倒计时

        Args:
            minutes: 倒计时分钟数
            tick_callback: 每分钟回调，参数为剩余分钟数
            finished_callback: 倒计时结束回调

        Returns:
            倒计时线程
        """
        self._cancelled = False
        self._countdown_active = True

        def countdown_thread():
            remaining = minutes

            while remaining > 0 and not self._cancelled:
                logger.debug(f"关机倒计时: {remaining}分钟")

                if tick_callback:
                    try:
                        tick_callback(remaining)
                    except Exception as e:
                        logger.warning(f"倒计时回调失败: {e}")

                # 每分钟通知一次（最后5分钟每30秒通知一次）
                if remaining <= 5 or remaining == minutes:
                    self._show_notification(
                        title=self._config.title,
                        message=f"系统将在 {remaining} 分钟后关机",
                        duration=5
                    )

                # 等待1分钟或直到取消
                for _ in range(60):
                    if self._cancelled:
                        break
                    time.sleep(1)

                remaining -= 1

            self._countdown_active = False

            if not self._cancelled and finished_callback:
                try:
                    finished_callback()
                except Exception as e:
                    logger.warning(f"倒计时结束回调失败: {e}")

        self._notification_thread = threading.Thread(
            target=countdown_thread,
            daemon=True
        )
        self._notification_thread.start()

        return self._notification_thread

    def cancel_countdown(self) -> bool:
        """
        取消倒计时

        Returns:
            是否成功取消
        """
        if not self._countdown_active:
            return False

        self._cancelled = True
        logger.info("倒计时已取消")

        # 发送取消通知
        self.notify_shutdown_cancelled()

        return True

    def is_countdown_active(self) -> bool:
        """
        检查倒计时是否正在进行

        Returns:
            是否正在倒计时
        """
        return self._countdown_active

    def _show_notification(
        self,
        title: str,
        message: str,
        icon_path: Optional[str] = None,
        duration: int = 10
    ) -> bool:
        """
        显示通知

        Args:
            title: 通知标题
            message: 通知消息
            icon_path: 图标路径
            duration: 显示时长（秒）

        Returns:
            是否成功显示
        """
        try:
            if self._toast is not None:
                self._toast.show_toast(
                    title=title,
                    msg=message,
                    icon_path=icon_path,
                    duration=duration,
                    threaded=True
                )
                return True
            else:
                # 备用通知方式：使用ctypes调用Windows API
                return self._show_windows_notification(title, message)

        except Exception as e:
            logger.warning(f"显示通知失败: {e}")
            return self._show_windows_notification(title, message)

    def _show_windows_notification(self, title: str, message: str) -> bool:
        """
        使用Windows API显示通知（备用方式）

        Args:
            title: 通知标题
            message: 通知消息

        Returns:
            是否成功显示
        """
        try:
            import ctypes
            from ctypes import wintypes

            # 使用MessageBox作为备用通知方式
            # 在实际应用中，这里可以使用更复杂的Windows通知API
            MB_OK = 0x00000000
            MB_ICONINFORMATION = 0x00000040
            MB_SYSTEMMODAL = 0x00001000

            # 非阻塞式通知
            ctypes.windll.user32.MessageBoxW(
                0,
                message,
                title,
                MB_OK | MB_ICONINFORMATION
            )
            return True

        except Exception as e:
            logger.error(f"Windows通知失败: {e}")
            return False

    def update_config(self, config: NotificationConfig):
        """
        更新通知配置

        Args:
            config: 新的通知配置
        """
        self._config = config
        logger.info("通知配置已更新")

    def get_config(self) -> NotificationConfig:
        """
        获取当前通知配置

        Returns:
            当前配置
        """
        return self._config


class ConsoleNotifier(UserNotifier):
    """
    控制台通知器（用于测试或没有GUI的环境）
    """

    def __init__(self, config: Optional[NotificationConfig] = None):
        """
        Args:
            config: 通知配置
        """
        self._config = config or NotificationConfig()
        self._cancelled = False
        self._countdown_active = False
        self._last_notification_time: Optional[float] = None
        # 不初始化Toast
        self._toast = None

    def _show_notification(
        self,
        title: str,
        message: str,
        icon_path: Optional[str] = None,
        duration: int = 10
    ) -> bool:
        """
        在控制台显示通知

        Args:
            title: 通知标题
            message: 通知消息
            icon_path: 图标路径
            duration: 显示时长（秒）

        Returns:
            是否成功显示
        """
        print(f"\n{'=' * 50}")
        print(f"[{title}]")
        print(f"{message}")
        print(f"{'=' * 50}\n")
        return True

    def _init_toast(self):
        """重写初始化，不使用Toast"""
        self._toast = None
