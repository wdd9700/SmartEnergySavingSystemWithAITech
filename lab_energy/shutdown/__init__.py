"""
智能关机决策系统

Module-3J: 基于多因素分析的智能关机决策系统

功能:
- 多因素关机决策（进程、用户活动、时间策略）
- Windows Toast通知
- 倒计时和取消机制
- 安全关机流程

示例:
    >>> from lab_energy.shutdown import ShutdownDecisionEngine, ShutdownExecutor
    >>> from lab_energy.scanner import ProcessScanner
    >>> 
    >>> # 创建组件
    >>> scanner = ProcessScanner()
    >>> engine = ShutdownDecisionEngine(scanner)
    >>> executor = ShutdownExecutor(engine)
    >>> 
    >>> # 评估决策
    >>> result = engine.evaluate()
    >>> print(f"决策: {result.decision.value}")
    >>> 
    >>> # 执行关机流程
    >>> executor.run_shutdown_flow()
"""

from .decision_engine import (
    ShutdownDecision,
    RiskLevel,
    ShutdownContext,
    DecisionResult,
    ShutdownDecisionEngine,
)

from .user_notifier import (
    NotificationType,
    NotificationConfig,
    UserNotifier,
    ConsoleNotifier,
)

from .shutdown_executor import (
    ShutdownExecutor,
    ScheduledShutdownExecutor,
)

__version__ = "1.0.0"
__author__ = "Lab Energy Team"

__all__ = [
    # 决策引擎
    "ShutdownDecision",
    "RiskLevel",
    "ShutdownContext",
    "DecisionResult",
    "ShutdownDecisionEngine",
    # 用户通知
    "NotificationType",
    "NotificationConfig",
    "UserNotifier",
    "ConsoleNotifier",
    # 关机执行器
    "ShutdownExecutor",
    "ScheduledShutdownExecutor",
]
