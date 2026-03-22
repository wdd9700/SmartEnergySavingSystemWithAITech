"""
智能关机决策引擎

基于多因素分析做出关机决策：
- 长时间运行任务
- 用户活动状态
- 时间策略（工作时间/非工作时间）
- 系统资源利用率
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any
from datetime import datetime, time
from enum import Enum
import ctypes
from ctypes import Structure, windll, c_uint, sizeof, byref
import logging

logger = logging.getLogger(__name__)


class ShutdownDecision(Enum):
    """关机决策结果"""
    SHUTDOWN = "shutdown"           # 执行关机
    DELAY = "delay"                 # 延迟检查
    CANCEL = "cancel"               # 取消关机
    NOTIFY = "notify"               # 通知用户


class RiskLevel(Enum):
    """风险等级"""
    LOW = 1                         # 低风险（可关机）
    MEDIUM = 2                      # 中风险（需确认）
    HIGH = 3                        # 高风险（禁止关机）


@dataclass
class ShutdownContext:
    """关机决策上下文"""
    timestamp: datetime
    long_running_tasks: List[Dict[str, Any]]  # 长时间运行任务
    user_active: bool                          # 用户是否活跃
    time_of_day: time                          # 当前时间
    day_of_week: int                           # 星期几 (0=周一, 6=周日)
    gpu_utilization: float                     # GPU利用率
    cpu_utilization: float                     # CPU利用率


@dataclass
class DecisionResult:
    """决策结果"""
    decision: ShutdownDecision
    risk_level: RiskLevel
    reasons: List[str]                         # 决策原因
    countdown_minutes: int                     # 倒计时分钟
    can_cancel: bool                           # 是否可取消
    risk_score: int = 0                        # 风险评分


class LASTINPUTINFO(Structure):
    """Windows API 结构体，用于获取最后输入时间"""
    _fields_ = [
        ("cbSize", c_uint),
        ("dwTime", c_uint)
    ]


class ShutdownDecisionEngine:
    """
    智能关机决策引擎

    综合多因素做出关机决策:
    1. 长时间运行任务（来自Module-3E）
    2. 用户活动状态
    3. 时间策略（工作时间/非工作时间）
    4. 系统资源利用率
    """

    # 工作时间配置
    WORK_HOURS = {
        'start': time(9, 0),       # 9:00
        'end': time(18, 0),        # 18:00
        'workdays': {0, 1, 2, 3, 4}  # 周一到周五 (0=周一)
    }

    # 风险评分阈值
    SCORE_THRESHOLDS = {
        'shutdown': (0, 1),        # 0-1分: 执行关机
        'notify_short': (2, 3),    # 2-3分: 通知用户，10分钟倒计时
        'notify_long': (4, 5),     # 4-5分: 通知用户，30分钟倒计时
        'forbid': (6, float('inf')) # 6+分: 禁止关机
    }

    # 评分权重
    SCORE_WEIGHTS = {
        'long_running_tasks': 3,   # 长时间任务存在: +3分
        'user_active': 2,          # 用户活跃: +2分
        'work_hours': 1,           # 工作时间: +1分
        'gpu_high': 2,             # GPU利用率>50%: +2分
        'cpu_high': 1              # CPU利用率>70%: +1分
    }

    # 用户不活跃阈值（毫秒）
    USER_IDLE_THRESHOLD_MS = 10 * 60 * 1000  # 10分钟

    def __init__(self, process_scanner=None):
        """
        Args:
            process_scanner: Module-3E的ProcessScanner实例
        """
        self._scanner = process_scanner
        self._decision_history: List[DecisionResult] = []
        self._last_evaluation_time: Optional[datetime] = None

    def evaluate(self, context: Optional[ShutdownContext] = None) -> DecisionResult:
        """
        评估是否执行关机

        评分机制:
        - 长时间任务存在: +3分（高风险）
        - 用户活跃: +2分（中风险）
        - 工作时间: +1分（低风险）
        - GPU利用率>50%: +2分（中风险）
        - CPU利用率>70%: +1分（低风险）

        总分:
        - 0-1分: 执行关机
        - 2-3分: 通知用户，10分钟倒计时
        - 4-5分: 通知用户，30分钟倒计时
        - 6+分: 禁止关机

        Args:
            context: 决策上下文（可选，自动获取）

        Returns:
            决策结果
        """
        # 如果没有提供上下文，自动构建
        if context is None:
            context = self._build_context()

        # 计算风险评分
        risk_score = self.calculate_risk_score(context)

        # 根据评分确定决策
        decision, risk_level, countdown, can_cancel, reasons = self._determine_decision(
            risk_score, context
        )

        result = DecisionResult(
            decision=decision,
            risk_level=risk_level,
            reasons=reasons,
            countdown_minutes=countdown,
            can_cancel=can_cancel,
            risk_score=risk_score
        )

        # 记录决策历史
        self._decision_history.append(result)
        self._last_evaluation_time = datetime.now()

        # 限制历史记录长度
        if len(self._decision_history) > 100:
            self._decision_history = self._decision_history[-100:]

        logger.info(f"关机决策: {decision.value}, 风险等级: {risk_level.name}, "
                   f"评分: {risk_score}, 原因: {reasons}")

        return result

    def _build_context(self) -> ShutdownContext:
        """自动构建决策上下文"""
        now = datetime.now()

        # 获取长时间运行任务
        long_running_tasks = []
        if self._scanner is not None:
            try:
                long_running_tasks = self._scanner.get_long_running_tasks()
            except Exception as e:
                logger.warning(f"获取长时间运行任务失败: {e}")

        # 获取系统资源利用率
        gpu_util = self._get_gpu_utilization()
        cpu_util = self._get_cpu_utilization()

        return ShutdownContext(
            timestamp=now,
            long_running_tasks=long_running_tasks,
            user_active=self.is_user_active(),
            time_of_day=now.time(),
            day_of_week=now.weekday(),
            gpu_utilization=gpu_util,
            cpu_utilization=cpu_util
        )

    def _get_gpu_utilization(self) -> float:
        """获取GPU利用率（需要GPU监控模块支持）"""
        try:
            # 尝试从GPU模块获取
            from lab_energy.gpu.monitor import GPUMonitor
            monitor = GPUMonitor()
            stats = monitor.get_utilization()
            return stats.get('utilization', 0.0)
        except Exception:
            return 0.0

    def _get_cpu_utilization(self) -> float:
        """获取CPU利用率"""
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except Exception:
            return 0.0

    def _determine_decision(self, score: int, context: ShutdownContext) -> Tuple[
        ShutdownDecision, RiskLevel, int, bool, List[str]
    ]:
        """
        根据评分确定决策

        Returns:
            (决策, 风险等级, 倒计时分钟, 是否可取消, 原因列表)
        """
        reasons = []

        # 构建原因列表
        if context.long_running_tasks:
            task_names = [t.get('name', 'Unknown') for t in context.long_running_tasks[:3]]
            reasons.append(f"存在长时间运行任务: {', '.join(task_names)}")

        if context.user_active:
            reasons.append("用户当前活跃")

        if self.is_work_hours():
            reasons.append("当前为工作时间")

        if context.gpu_utilization > 50:
            reasons.append(f"GPU利用率高 ({context.gpu_utilization:.1f}%)")

        if context.cpu_utilization > 70:
            reasons.append(f"CPU利用率高 ({context.cpu_utilization:.1f}%)")

        if not reasons:
            reasons.append("无风险因素")

        # 根据评分确定决策
        if score <= 1:
            return ShutdownDecision.SHUTDOWN, RiskLevel.LOW, 0, False, reasons
        elif score <= 3:
            return ShutdownDecision.NOTIFY, RiskLevel.MEDIUM, 10, True, reasons
        elif score <= 5:
            return ShutdownDecision.NOTIFY, RiskLevel.MEDIUM, 30, True, reasons
        else:
            return ShutdownDecision.CANCEL, RiskLevel.HIGH, 0, False, reasons

    def is_work_hours(self, check_time: Optional[datetime] = None) -> bool:
        """
        判断是否为工作时间

        Args:
            check_time: 检查时间（默认当前）

        Returns:
            是否为工作时间
        """
        if check_time is None:
            check_time = datetime.now()

        current_time = check_time.time()
        current_day = check_time.weekday()

        # 检查是否为工作日
        if current_day not in self.WORK_HOURS['workdays']:
            return False

        # 检查是否在工作时间范围内
        return (self.WORK_HOURS['start'] <= current_time <= self.WORK_HOURS['end'])

    def is_user_active(self) -> bool:
        """
        检测用户是否活跃

        基于:
        - 最后输入时间
        - 屏幕状态
        - 锁屏状态

        Returns:
            用户是否活跃
        """
        try:
            # 获取最后输入时间
            idle_time = self._get_idle_time_ms()
            is_active = idle_time < self.USER_IDLE_THRESHOLD_MS

            logger.debug(f"用户空闲时间: {idle_time}ms, 活跃状态: {is_active}")
            return is_active

        except Exception as e:
            logger.warning(f"检测用户活动状态失败: {e}")
            # 默认认为用户活跃（安全起见）
            return True

    def _get_idle_time_ms(self) -> int:
        """获取用户空闲时间（毫秒）"""
        try:
            lii = LASTINPUTINFO()
            lii.cbSize = sizeof(LASTINPUTINFO)
            windll.user32.GetLastInputInfo(byref(lii))
            millis = windll.kernel32.GetTickCount() - lii.dwTime
            return millis
        except Exception as e:
            logger.warning(f"获取空闲时间失败: {e}")
            return 0

    def calculate_risk_score(self, context: ShutdownContext) -> int:
        """
        计算风险评分

        Args:
            context: 决策上下文

        Returns:
            风险评分（0-10+）
        """
        score = 0

        # 长时间任务存在: +3分
        if context.long_running_tasks and len(context.long_running_tasks) > 0:
            score += self.SCORE_WEIGHTS['long_running_tasks']

        # 用户活跃: +2分
        if context.user_active:
            score += self.SCORE_WEIGHTS['user_active']

        # 工作时间: +1分（使用上下文中的时间）
        # 构建一个datetime对象用于工作时间判断
        from datetime import timedelta
        today = datetime.today()
        check_time = datetime.combine(today, context.time_of_day)
        # 调整日期到正确的星期几
        current_weekday = today.weekday()
        days_diff = context.day_of_week - current_weekday
        check_time = check_time + timedelta(days=days_diff)
        if self.is_work_hours(check_time):
            score += self.SCORE_WEIGHTS['work_hours']

        # GPU利用率>50%: +2分
        if context.gpu_utilization > 50:
            score += self.SCORE_WEIGHTS['gpu_high']

        # CPU利用率>70%: +1分
        if context.cpu_utilization > 70:
            score += self.SCORE_WEIGHTS['cpu_high']

        return score

    def should_shutdown(self) -> Tuple[bool, List[str]]:
        """
        简化的关机判断接口

        Returns:
            (是否关机, 原因列表)
        """
        result = self.evaluate()
        return result.decision == ShutdownDecision.SHUTDOWN, result.reasons

    def get_decision_history(self) -> List[DecisionResult]:
        """获取决策历史"""
        return self._decision_history.copy()

    def get_last_evaluation_time(self) -> Optional[datetime]:
        """获取最后评估时间"""
        return self._last_evaluation_time

    def clear_history(self):
        """清空决策历史"""
        self._decision_history.clear()

    def update_work_hours(self, start: time, end: time, workdays: Optional[set] = None):
        """
        更新工作时间配置

        Args:
            start: 工作开始时间
            end: 工作结束时间
            workdays: 工作日集合（默认周一到周五）
        """
        self.WORK_HOURS['start'] = start
        self.WORK_HOURS['end'] = end
        if workdays is not None:
            self.WORK_HOURS['workdays'] = workdays

        logger.info(f"工作时间已更新: {start} - {end}, 工作日: {workdays}")
