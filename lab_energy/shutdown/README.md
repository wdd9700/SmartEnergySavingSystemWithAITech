# Module-3J: 智能关机决策系统

基于多因素分析的智能关机决策系统，用于实验室能源管理。

## 功能特性

- **多因素决策**：综合进程、用户活动、时间策略、系统资源等因素
- **风险评分机制**：0-10+ 分评分系统，科学决策
- **用户通知**：Windows Toast 通知 + 控制台通知
- **倒计时机制**：支持 10分钟/30分钟 倒计时
- **取消机制**：用户可随时取消关机
- **定时评估**：支持每 10 分钟自动评估
- **工作时间配置**：可自定义工作时间和工作日

## 安装依赖

```bash
# 可选：安装 Windows Toast 通知支持
pip install win10toast

# 必需：系统监控
pip install psutil
```

## 快速开始

### 基本使用

```python
from lab_energy.shutdown import ShutdownDecisionEngine, ShutdownExecutor

# 创建决策引擎
engine = ShutdownDecisionEngine()

# 评估当前状态
result = engine.evaluate()
print(f"决策: {result.decision.value}")
print(f"风险等级: {result.risk_level.name}")
print(f"原因: {result.reasons}")

# 执行关机流程（如果需要）
executor = ShutdownExecutor(engine)
executor.run_shutdown_flow()
```

### 使用自定义上下文

```python
from datetime import datetime, time
from lab_energy.shutdown import ShutdownDecisionEngine, ShutdownContext

engine = ShutdownDecisionEngine()

# 创建自定义上下文
context = ShutdownContext(
    timestamp=datetime.now(),
    long_running_tasks=[{"name": "ML Training", "pid": 1234}],
    user_active=True,
    time_of_day=time(10, 0),
    day_of_week=0,  # 周一
    gpu_utilization=80.0,
    cpu_utilization=90.0
)

result = engine.evaluate(context)
```

### 定时关机

```python
from lab_energy.shutdown import ShutdownDecisionEngine, ScheduledShutdownExecutor

engine = ShutdownDecisionEngine()
scheduler = ScheduledShutdownExecutor(
    decision_engine=engine,
    evaluation_interval=600  # 每10分钟评估一次
)

# 启动定时评估
scheduler.start_scheduler()

# ... 运行一段时间后 ...

# 停止定时评估
scheduler.stop_scheduler()
```

## 决策评分机制

| 风险因素 | 分值 | 说明 |
|---------|------|------|
| 长时间任务存在 | +3 | 保护长时间运行任务 |
| 用户活跃 | +2 | 检测键盘鼠标活动 |
| 工作时间 | +1 | 工作时间（默认 9:00-18:00） |
| GPU利用率>50% | +2 | 高GPU负载 |
| CPU利用率>70% | +1 | 高CPU负载 |

### 决策阈值

| 总分 | 决策 | 倒计时 | 可取消 |
|------|------|--------|--------|
| 0-1 | 执行关机 | 0分钟 | 否 |
| 2-3 | 通知用户 | 10分钟 | 是 |
| 4-5 | 通知用户 | 30分钟 | 是 |
| 6+ | 禁止关机 | - | - |

## 模块结构

```
lab_energy/shutdown/
├── __init__.py              # 模块导出
├── decision_engine.py       # 决策引擎
├── user_notifier.py         # 用户通知
├── shutdown_executor.py     # 关机执行器
├── example.py               # 使用示例
├── README.md                # 本文档
└── tests/
    ├── __init__.py
    └── test_shutdown.py     # 单元测试
```

## API 文档

### ShutdownDecisionEngine

决策引擎类，负责评估是否执行关机。

#### 方法

- `evaluate(context=None) -> DecisionResult`: 评估是否执行关机
- `is_work_hours(check_time=None) -> bool`: 判断是否为工作时间
- `is_user_active() -> bool`: 检测用户是否活跃
- `calculate_risk_score(context) -> int`: 计算风险评分
- `should_shutdown() -> tuple[bool, List[str]]`: 简化的关机判断
- `update_work_hours(start, end, workdays=None)`: 更新工作时间配置

### ShutdownExecutor

关机执行器类，负责执行关机流程。

#### 方法

- `run_shutdown_flow() -> bool`: 执行完整的关机流程
- `notify_user(message, timeout_minutes) -> bool`: 通知用户
- `countdown(minutes, callback=None) -> bool`: 倒计时等待
- `cancel_shutdown()`: 取消关机
- `save_user_work() -> bool`: 尝试保存用户工作
- `execute_shutdown(force=False) -> bool`: 执行系统关机
- `abort_shutdown() -> bool`: 中止已发起的关机命令

### ScheduledShutdownExecutor

定时关机执行器，支持定时评估和自动关机。

#### 方法

- `start_scheduler()`: 启动定时评估调度器
- `stop_scheduler()`: 停止定时评估调度器
- `is_running() -> bool`: 检查调度器是否正在运行
- `update_interval(interval)`: 更新评估间隔

## 运行测试

```bash
# 运行所有测试
python -m pytest lab_energy/shutdown/tests/test_shutdown.py -v

# 或使用 unittest
python -m unittest lab_energy.shutdown.tests.test_shutdown -v
```

## 运行示例

```bash
# 设置 PYTHONPATH
$env:PYTHONPATH="$env:PYTHONPATH;e:\projects\Coding\SmartEnergySavinginLightControlandACControl"

# 运行示例
python lab_energy/shutdown/example.py
```

## 注意事项

1. **管理员权限**：执行关机命令需要管理员权限
2. **长时间任务保护**：系统会严格保护长时间运行任务，防止误关机
3. **用户取消**：倒计时期间用户可以随时取消关机
4. **Windows Toast**：如需使用 Windows Toast 通知，请安装 `win10toast` 库

## 许可证

MIT License
