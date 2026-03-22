"""
智能关机决策系统使用示例

演示如何使用 Module-3J 智能关机决策系统
"""

import logging
from datetime import datetime, time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from lab_energy.shutdown import (
    ShutdownDecisionEngine,
    ShutdownExecutor,
    ConsoleNotifier,
    ShutdownContext,
    ScheduledShutdownExecutor,
)


def example_basic_usage():
    """基本使用示例"""
    print("\n" + "=" * 60)
    print("示例 1: 基本使用")
    print("=" * 60)

    # 创建决策引擎
    engine = ShutdownDecisionEngine()

    # 评估当前状态
    result = engine.evaluate()

    print(f"决策结果: {result.decision.value}")
    print(f"风险等级: {result.risk_level.name}")
    print(f"风险评分: {result.risk_score}")
    print(f"倒计时: {result.countdown_minutes} 分钟")
    print(f"可取消: {result.can_cancel}")
    print(f"原因: {result.reasons}")


def example_with_context():
    """使用自定义上下文示例"""
    print("\n" + "=" * 60)
    print("示例 2: 使用自定义上下文")
    print("=" * 60)

    engine = ShutdownDecisionEngine()

    # 创建自定义上下文（模拟周末晚上，无任务）
    now = datetime.now()
    context = ShutdownContext(
        timestamp=now,
        long_running_tasks=[],  # 无长时间任务
        user_active=False,       # 用户不活跃
        time_of_day=time(20, 0), # 晚上8点
        day_of_week=5,           # 周六
        gpu_utilization=30.0,    # GPU低利用率
        cpu_utilization=50.0     # CPU中等利用率
    )

    result = engine.evaluate(context)

    print(f"决策结果: {result.decision.value}")
    print(f"风险等级: {result.risk_level.name}")
    print(f"风险评分: {result.risk_score}")
    print(f"原因: {result.reasons}")


def example_high_risk():
    """高风险场景示例"""
    print("\n" + "=" * 60)
    print("示例 3: 高风险场景（工作时间，有长时间任务）")
    print("=" * 60)

    engine = ShutdownDecisionEngine()

    # 模拟工作时间，有长时间任务
    now = datetime.now()
    context = ShutdownContext(
        timestamp=now,
        long_running_tasks=[
            {"name": "ML Training", "pid": 1234, "runtime": 3600},
            {"name": "Data Processing", "pid": 5678, "runtime": 1800},
        ],
        user_active=True,        # 用户活跃
        time_of_day=time(10, 0), # 上午10点
        day_of_week=0,           # 周一
        gpu_utilization=80.0,    # GPU高利用率
        cpu_utilization=90.0     # CPU高利用率
    )

    result = engine.evaluate(context)

    print(f"决策结果: {result.decision.value}")
    print(f"风险等级: {result.risk_level.name}")
    print(f"风险评分: {result.risk_score}")
    print(f"原因: {result.reasons}")


def example_shutdown_executor():
    """关机执行器示例"""
    print("\n" + "=" * 60)
    print("示例 4: 关机执行器（使用控制台通知器）")
    print("=" * 60)

    # 创建决策引擎和执行器
    engine = ShutdownDecisionEngine()
    executor = ShutdownExecutor(
        decision_engine=engine,
        use_console_notifier=True  # 使用控制台通知
    )

    # 添加回调函数
    def on_shutdown():
        print("[回调] 系统即将关机...")

    def on_cancel():
        print("[回调] 关机已取消")

    executor.add_shutdown_callback(on_shutdown)
    executor.add_cancel_callback(on_cancel)

    # 检查是否可以关机
    should_shutdown, reasons = engine.should_shutdown()
    print(f"是否可以关机: {should_shutdown}")
    print(f"原因: {reasons}")

    # 注意：这里不实际执行关机，仅演示
    print("\n提示: 实际使用时调用 executor.run_shutdown_flow()")


def example_work_hours():
    """工作时间判断示例"""
    print("\n" + "=" * 60)
    print("示例 5: 工作时间判断")
    print("=" * 60)

    engine = ShutdownDecisionEngine()

    # 测试不同时间
    test_times = [
        (datetime(2024, 1, 1, 10, 0), "周一上午10点"),   # 工作时间
        (datetime(2024, 1, 1, 20, 0), "周一晚上8点"),    # 非工作时间
        (datetime(2024, 1, 6, 10, 0), "周六上午10点"),   # 周末
        (datetime(2024, 1, 1, 8, 0), "周一上午8点"),     # 上班前
    ]

    for test_time, description in test_times:
        is_work = engine.is_work_hours(test_time)
        print(f"{description}: {'工作时间' if is_work else '非工作时间'}")


def example_custom_work_hours():
    """自定义工作时间示例"""
    print("\n" + "=" * 60)
    print("示例 6: 自定义工作时间")
    print("=" * 60)

    engine = ShutdownDecisionEngine()

    # 更新工作时间为 8:30 - 17:30，周一到周六
    engine.update_work_hours(
        start=time(8, 30),
        end=time(17, 30),
        workdays={0, 1, 2, 3, 4, 5}  # 周一到周六
    )

    test_time = datetime(2024, 1, 6, 10, 0)  # 周六上午10点
    is_work = engine.is_work_hours(test_time)
    print(f"周六上午10点: {'工作时间' if is_work else '非工作时间'}")

    test_time = datetime(2024, 1, 7, 10, 0)  # 周日上午10点
    is_work = engine.is_work_hours(test_time)
    print(f"周日上午10点: {'工作时间' if is_work else '非工作时间'}")


def example_scheduled_shutdown():
    """定时关机示例"""
    print("\n" + "=" * 60)
    print("示例 7: 定时关机调度器")
    print("=" * 60)

    engine = ShutdownDecisionEngine()
    scheduler = ScheduledShutdownExecutor(
        decision_engine=engine,
        evaluation_interval=600  # 每10分钟评估一次
    )

    print(f"评估间隔: {scheduler._evaluation_interval} 秒")
    print(f"调度器运行状态: {scheduler.is_running()}")

    # 注意：实际使用时调用 scheduler.start_scheduler()
    print("\n提示: 实际使用时调用 scheduler.start_scheduler() 启动定时评估")
    print("      调用 scheduler.stop_scheduler() 停止定时评估")


if __name__ == "__main__":
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("智能关机决策系统使用示例")
    print("=" * 60)

    example_basic_usage()
    example_with_context()
    example_high_risk()
    example_shutdown_executor()
    example_work_hours()
    example_custom_work_hours()
    example_scheduled_shutdown()

    print("\n" + "=" * 60)
    print("所有示例运行完成")
    print("=" * 60)
