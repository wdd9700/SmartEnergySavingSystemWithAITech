# 方向三：计算机节能系统（实验室场景）- 技术栈调研报告

## 调研日期
2026年3月19日

---

## 1. 系统监控技术栈

### 1.1 进程监控 - psutil (强烈推荐)
**项目地址**: https://github.com/giampaolo/psutil  
**文档**: https://psutil.readthedocs.io/  
**版本**: 6.0+ (2024年发布)

**核心功能**:
- 跨平台进程监控（Windows/Linux/macOS/FreeBSD）
- CPU/内存/磁盘/网络/传感器监控
- 进程列表获取、进程信息查询
- 进程启动/终止/暂停/恢复
- 系统负载监控和进程优先级管理
- **新增**: 更好的ARM64支持，性能优化

**性能优化示例**:
```python
import psutil
from functools import lru_cache

class ProcessMonitor:
    """高性能进程监控器"""
    
    def __init__(self):
        self._process_cache = {}
        self._last_update = 0
    
    @lru_cache(maxsize=128)
    def get_process_info(self, pid):
        """缓存进程信息，减少系统调用"""
        try:
            p = psutil.Process(pid)
            return {
                'name': p.name(),
                'cpu_percent': p.cpu_percent(interval=0.1),
                'memory_percent': p.memory_percent(),
                'status': p.status(),
                'create_time': p.create_time()
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None
    
    def get_all_processes(self, filter_names=None):
        """高效获取进程列表"""
        processes = []
        # 使用一次系统调用获取所有进程
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                info = proc.info
                if filter_names and info['name'] not in filter_names:
                    continue
                processes.append(info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return processes

# 使用示例
monitor = ProcessMonitor()
processes = monitor.get_all_processes(filter_names=['chrome.exe', 'python.exe'])
```

**适用场景**:
- ✅ 定时扫描进程列表（每秒一次无压力）
- ✅ 监控进程资源占用
- ✅ 判断进程是否为长时间运行任务
- ✅ 进程优先级动态调整

---

### 1.2 GPU监控 (推荐方案更新)

#### pynvml - NVIDIA官方库 (强烈推荐)
**项目地址**: https://github.com/NVIDIA/pynvml  
**优势**: 官方维护，功能完整，性能最优

```python
from pynvml import *

class GPUMonitor:
    """高性能GPU监控器"""
    
    def __init__(self):
        nvmlInit()
        self.device_count = nvmlDeviceGetCount()
        self.handles = [nvmlDeviceGetHandleByIndex(i) for i in range(self.device_count)]
    
    def get_gpu_info(self, gpu_id=0):
        """获取GPU详细信息"""
        handle = self.handles[gpu_id]
        
        # 利用率
        util = nvmlDeviceGetUtilizationRates(handle)
        
        # 显存
        mem = nvmlDeviceGetMemoryInfo(handle)
        
        # 功耗
        power = nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
        
        # 温度
        temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
        
        # 时钟频率
        clock = nvmlDeviceGetClockInfo(handle, NVML_CLOCK_GRAPHICS)
        
        return {
            'gpu_id': gpu_id,
            'name': nvmlDeviceGetName(handle),
            'gpu_util': util.gpu,
            'memory_util': util.memory,
            'memory_used': mem.used / 1024**2,  # MB
            'memory_total': mem.total / 1024**2,  # MB
            'power': power,
            'temperature': temp,
            'clock_mhz': clock
        }
    
    def set_power_limit(self, gpu_id, watts):
        """设置GPU功耗限制 (需要管理员权限)"""
        handle = self.handles[gpu_id]
        nvmlDeviceSetPowerManagementLimit(handle, watts * 1000)  # 转换为mW
    
    def shutdown(self):
        nvmlShutdown()

# 使用示例
monitor = GPUMonitor()
info = monitor.get_gpu_info(0)
print(f"GPU: {info['name']}, 功耗: {info['power']:.1f}W, 温度: {info['temperature']}°C")
```

#### GPUtil - 简化版 (备选)
**项目地址**: https://github.com/anderskm/gputil

```python
import GPUtil

# 快速获取GPU信息 (基于pynvml的封装)
gpus = GPUtil.getGPUs()
for gpu in gpus:
    print(f"GPU {gpu.id}: {gpu.name}")
    print(f"  Load: {gpu.load*100}%")
    print(f"  Memory: {gpu.memoryUsed}/{gpu.memoryTotal} MB")
    print(f"  Temperature: {gpu.temperature}°C")
```

**选型建议**:
- ✅ **pynvml**: 需要完整功能（功耗控制、时钟调节）时使用
- ✅ **GPUtil**: 仅需监控信息，追求代码简洁时使用

---

## 2. Windows电源管理技术栈

### 2.1 Windows电源方案管理 (优化方案)

**高性能Python封装**:
```python
import subprocess
import ctypes
from ctypes import wintypes
from enum import Enum

class PowerPlan(Enum):
    """Windows电源方案GUID"""
    BALANCED = "381b4222-f694-41f0-9685-ff5bb260df2e"
    HIGH_PERFORMANCE = "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"
    POWER_SAVER = "a1841308-3541-4fab-bc81-f71556f20b4a"
    ULTIMATE_PERFORMANCE = "e9a42b02-d5df-448d-aa00-03f14749eb61"  # Win10专业版+

class WindowsPowerManager:
    """Windows电源管理高级封装"""
    
    def __init__(self):
        self.kernel32 = ctypes.windll.kernel32
        self.powrprof = ctypes.windll.powrprof
        
    def set_power_plan(self, plan: PowerPlan):
        """切换电源方案"""
        subprocess.run(['powercfg', '/setactive', plan.value], check=True)
        
    def get_current_plan(self) -> str:
        """获取当前电源方案"""
        result = subprocess.run(['powercfg', '/getactivescheme'], 
                              capture_output=True, text=True, check=True)
        return result.stdout.strip()
    
    def set_cpu_throttle(self, max_percent: int, ac_power: bool = True):
        """设置CPU最大性能状态 (0-100)"""
        power_type = 'ac' if ac_power else 'dc'
        
        # 处理器电源管理GUID
        SUB_PROCESSOR = "54533251-82be-4824-96c1-47b60b740d00"
        PROC_THROTTLE_MAX = "bc5038f7-23e0-4960-96da-33abaf5935ec"
        
        subprocess.run([
            'powercfg', f'/set{power_type}valueindex', 'scheme_current',
            SUB_PROCESSOR, PROC_THROTTLE_MAX, str(max_percent)
        ], check=True)
        subprocess.run(['powercfg', '/setactive', 'scheme_current'], check=True)
    
    def prevent_sleep(self):
        """阻止系统进入睡眠 (执行期间)"""
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ES_DISPLAY_REQUIRED = 0x00000002
        self.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)
    
    def allow_sleep(self):
        """允许系统进入睡眠"""
        ES_CONTINUOUS = 0x80000000
        self.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    
    def hibernate(self):
        """进入休眠状态"""
        subprocess.run(['rundll32.exe', 'powrprof.dll,SetSuspendState', 'Hibernate'])
    
    def sleep(self):
        """进入睡眠状态"""
        subprocess.run(['rundll32.exe', 'powrprof.dll,SetSuspendState', '0,1,0'])

# 使用示例
pm = WindowsPowerManager()
pm.set_power_plan(PowerPlan.POWER_SAVER)  # 切换到节能模式
pm.set_cpu_throttle(50)  # 限制CPU最大50%性能
```

**电源方案GUID**:
- 平衡: `381b4222-f694-41f0-9685-ff5bb260df2e`
- 高性能: `8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c`
- 节能: `a1841308-3541-4fab-bc81-f71556f20b4a`
- 卓越性能: `e9a42b02-d5df-448d-aa00-03f14749eb61` (Win10专业工作站版+)

### 2.2 WMI (Windows Management Instrumentation)

```python
import wmi

c = wmi.WMI()

# 获取电源信息
for battery in c.Win32_Battery():
    print(f"Battery: {battery.EstimatedChargeRemaining}%")

# 获取处理器信息
for processor in c.Win32_Processor():
    print(f"CPU: {processor.Name}")
    print(f"Load: {processor.LoadPercentage}%")

# 获取系统电源状态
for power in c.Win32_PowerPlan():
    if power.IsActive:
        print(f"Active Plan: {power.ElementName}")
```

### 2.3 Win32 API (ctypes)

```python
import ctypes
from ctypes import wintypes

# 设置系统关闭权限
def set_shutdown_privilege():
    """获取关机权限"""
    advapi32 = ctypes.windll.advapi32
    kernel32 = ctypes.windll.kernel32
    
    # 获取当前进程令牌
    hToken = wintypes.HANDLE()
    kernel32.OpenProcessToken(kernel32.GetCurrentProcess(), 0x0020 | 0x0008, ctypes.byref(hToken))
    
    # 查找关机权限
    luid = wintypes.LUID()
    advapi32.LookupPrivilegeValueW(None, "SeShutdownPrivilege", ctypes.byref(luid))
    
    # 启用权限
    tp = wintypes.TOKEN_PRIVILEGES()
    tp.PrivilegeCount = 1
    tp.Privileges[0].Luid = luid
    tp.Privileges[0].Attributes = 0x00000002  # SE_PRIVILEGE_ENABLED
    
    advapi32.AdjustTokenPrivileges(hToken, False, ctypes.byref(tp), 0, None, None)

# 执行关机
def shutdown_system():
    """执行系统关机"""
    set_shutdown_privilege()
    user32 = ctypes.windll.user32
    # EWX_SHUTDOWN | EWX_FORCE (0x01 | 0x04)
    user32.ExitWindowsEx(0x05, 0x00000000)
```

---

## 3. CPU/GPU频率控制技术栈

### 3.1 CPU频率控制

**Windows**: 通过电源方案设置处理器状态
```python
import subprocess

def set_cpu_max_state(percentage, ac_power=True):
    """
    设置CPU最大状态百分比
    percentage: 0-100
    ac_power: True=交流电源, False=电池
    """
    power_type = 'ac' if ac_power else 'dc'
    
    # 设置最大处理器状态
    subprocess.run([
        'powercfg', '/setacvalueindex', 'scheme_current', 
        '54533251-82be-4824-96c1-47b60b740d00',  # 处理器电源管理
        'bc5038f7-23e0-4960-96da-33abaf5935ec',  # 最大处理器状态
        str(percentage)
    ], check=True)
    
    # 应用设置
    subprocess.run(['powercfg', '/setactive', 'scheme_current'])

def set_cpu_min_state(percentage, ac_power=True):
    """设置CPU最小状态百分比"""
    power_type = 'ac' if ac_power else 'dc'
    
    subprocess.run([
        'powercfg', '/setacvalueindex', 'scheme_current',
        '54533251-82be-4824-96c1-47b60b740d00',
        '893dee8e-2bef-41e0-89c6-b55d0929964c',  # 最小处理器状态
        str(percentage)
    ], check=True)
    
    subprocess.run(['powercfg', '/setactive', 'scheme_current'])
```

**Linux**: cpufrequtils
```python
import subprocess

def set_cpu_governor(governor='powersave'):
    """
    设置CPU调度策略
    governor: powersave, performance, ondemand, conservative
    """
    # 获取CPU核心数
    with open('/proc/cpuinfo') as f:
        cpu_count = sum(1 for line in f if line.startswith('processor'))
    
    # 为每个核心设置调度策略
    for i in range(cpu_count):
        subprocess.run([
            'sudo', 'cpufreq-set', '-c', str(i), '-g', governor
        ])

def set_cpu_frequency(frequency_mhz):
    """设置CPU频率（需要root权限）"""
    subprocess.run([
        'sudo', 'cpufreq-set', '-f', f'{frequency_mhz}MHz'
    ])
```

### 3.2 NVIDIA GPU频率控制

```python
import subprocess

def set_gpu_power_limit(gpu_id, power_limit_watts):
    """
    设置GPU功耗限制
    gpu_id: GPU编号
    power_limit_watts: 功耗限制（瓦特）
    """
    subprocess.run([
        'nvidia-smi', '-i', str(gpu_id), 
        '-pl', str(power_limit_watts)
    ], check=True)

def set_gpu_clock_offset(gpu_id, offset_mhz):
    """
    设置GPU时钟偏移
    """
    subprocess.run([
        'nvidia-smi', '-i', str(gpu_id),
        '-lgc', f'{offset_mhz}'  # 锁定GPU时钟
    ], check=True)

def set_gpu_memory_clock_offset(gpu_id, offset_mhz):
    """设置显存时钟偏移"""
    subprocess.run([
        'nvidia-smi', '-i', str(gpu_id),
        '-lmc', f'{offset_mhz}'
    ], check=True)
```

---

## 4. LLM集成技术栈

### 4.1 OpenAI API

```python
import openai

client = openai.OpenAI(api_key="your-api-key")

def classify_process(process_name, cmdline, cpu_percent, memory_percent):
    """
    使用LLM判断进程类型和重要性
    """
    prompt = f"""
    分析以下进程，判断其类型和重要性：
    
    进程名: {process_name}
    命令行: {cmdline}
    CPU使用率: {cpu_percent}%
    内存使用率: {memory_percent}%
    
    请回答：
    1. 这是一个什么类型的进程？（系统进程/开发工具/AI训练/浏览器/其他）
    2. 是否可能是长时间运行的任务？（是/否/不确定）
    3. 是否可以安全关闭？（是/否/不确定）
    4. 建议操作：（保持运行/可以关闭/需要确认）
    
    以JSON格式返回。
    """
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "你是一个系统进程分析专家。"},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )
    
    return response.choices[0].message.content
```

### 4.2 本地LLM (Ollama)

```python
import requests
import json

def classify_process_local(process_name, cmdline, cpu_percent, memory_percent):
    """
    使用本地LLM判断进程类型
    """
    prompt = f"""
    分析以下进程，判断其类型和重要性：
    进程名: {process_name}
    命令行: {cmdline}
    CPU使用率: {cpu_percent}%
    内存使用率: {memory_percent}%
    """
    
    response = requests.post('http://localhost:11434/api/generate',
        json={
            'model': 'llama3',
            'prompt': prompt,
            'stream': False
        }
    )
    
    return response.json()['response']
```

---

## 5. 通知系统技术栈

### 5.1 Windows通知 (Win10toast)

```python
from win10toast import ToastNotifier

toaster = ToastNotifier()

def show_shutdown_notification(timeout_minutes=10):
    """显示关机通知"""
    toaster.show_toast(
        "智能节能系统",
        f"系统将在{timeout_minutes}分钟后关机。如需取消，请登录系统。",
        duration=30,
        threaded=True
    )
```

### 5.2 邮件通知

```python
import smtplib
from email.mime.text import MIMEText

def send_email_notification(to_email, subject, body):
    """发送邮件通知"""
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = 'lab-monitor@example.com'
    msg['To'] = to_email
    
    with smtplib.SMTP('smtp.example.com', 587) as server:
        server.starttls()
        server.login('username', 'password')
        server.send_message(msg)
```

### 5.3 WebSocket实时通知

```python
import asyncio
import websockets
import json

async def notify_user(websocket, message):
    """通过WebSocket发送通知"""
    await websocket.send(json.dumps({
        'type': 'shutdown_warning',
        'message': message,
        'timeout_minutes': 10
    }))
```

---

## 6. 定时任务调度技术栈

### 6.1 APScheduler (推荐)

```python
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

def check_and_shutdown():
    """检查进程并执行关机逻辑"""
    # 实现检查逻辑
    pass

# 创建调度器
scheduler = BackgroundScheduler()

# 每天晚上9点和10点执行检查
scheduler.add_job(
    check_and_shutdown,
    trigger=CronTrigger(hour='21,22', minute='0'),
    id='shutdown_check',
    name='智能关机检查'
)

# 每30分钟执行一次（晚上9-11点）
scheduler.add_job(
    check_and_shutdown,
    trigger=CronTrigger(hour='21-23', minute='*/30'),
    id='shutdown_check_frequent',
    name='频繁关机检查'
)

scheduler.start()
```

### 6.2 Windows任务计划程序

```python
import subprocess

def create_scheduled_task():
    """创建Windows计划任务"""
    # 使用schtasks命令创建任务
    subprocess.run([
        'schtasks', '/create',
        '/tn', 'SmartEnergyShutdown',
        '/tr', 'python C:\\path\\to\\shutdown_script.py',
        '/sc', 'daily',
        '/st', '21:00',
        '/f'
    ])
```

---

## 7. 推荐技术组合

### 7.1 完整技术栈

| 功能模块 | 推荐技术 | 备选方案 |
|---------|---------|---------|
| **进程监控** | psutil | wmic命令 |
| **GPU监控** | GPUtil + pynvml | nvidia-smi命令 |
| **电源管理** | powercfg + WMI | Windows API |
| **CPU频率控制** | powercfg | ThrottleStop |
| **GPU频率控制** | nvidia-smi | MSI Afterburner API |
| **LLM集成** | OpenAI API | Ollama本地模型 |
| **通知系统** | Win10toast + WebSocket | 邮件通知 |
| **任务调度** | APScheduler | Windows任务计划程序 |
| **配置管理** | YAML/JSON | 数据库 |

### 7.2 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    应用层 (Application)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ 进程分析器   │  │ 智能决策引擎 │  │ 关机控制器   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
├─────────────────────────────────────────────────────────────┤
│                    服务层 (Services)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ LLM服务      │  │ 通知服务     │  │ 调度服务     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
├─────────────────────────────────────────────────────────────┤
│                    硬件接口层 (Hardware Interface)            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ 进程监控     │  │ 电源管理     │  │ 频率控制     │      │
│  │ (psutil)     │  │ (powercfg)   │  │ (WMI/nvml)   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. 参考开源项目

| 项目 | 地址 | 说明 |
|-----|------|------|
| psutil | https://github.com/giampaolo/psutil | 跨平台系统监控 |
| GPUtil | https://github.com/anderskm/gputil | NVIDIA GPU监控 |
| pynvml | https://github.com/NVIDIA/pynvml | NVIDIA管理库 |
| APScheduler | https://github.com/agronholm/apscheduler | 任务调度 |
| win10toast | https://github.com/jithurjacob/Windows-10-Toast-Notifications | Windows通知 |
| pywin32 | https://github.com/mhammond/pywin32 | Windows API封装 |

---

## 9. 实施建议

### 9.1 开发优先级

1. **Phase 1**: 进程监控基础 (psutil)
2. **Phase 2**: 电源管理接口 (powercfg/WMI)
3. **Phase 3**: LLM集成 (进程分类)
4. **Phase 4**: 通知系统 (Toast/WebSocket)
5. **Phase 5**: 任务调度 (APScheduler)
6. **Phase 6**: 频率控制 (CPU/GPU)

### 9.2 注意事项

1. **权限问题**: 关机和电源管理需要管理员权限
2. **安全性**: LLM判断需要人工确认机制
3. **稳定性**: 关键进程（如系统进程）需要白名单保护
4. **用户体验**: 提供足够的通知时间和取消机制

---

## 文档版本
- 版本：v1.0
- 创建日期：2026年3月19日
- 状态：技术调研完成
