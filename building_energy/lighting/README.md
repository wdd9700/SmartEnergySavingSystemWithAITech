# 人因照明与预测性开关模块

## 模块概述

本模块实现基于昼夜节律的色温调节和基于运动预测的提前开灯策略，提升用户体验的同时保持节能效果。

## 核心功能

### 1. 人因照明 (Circadian Rhythm)

根据时间自动调节色温和亮度，符合人体昼夜节律：

| 时间段 | 色温 | 亮度 | 效果 |
|--------|------|------|------|
| 06:00-09:00 | 5000-6500K | 100% | 冷白光，提神醒脑 |
| 09:00-17:00 | 4000-5000K | 90% | 自然白光，保持专注 |
| 17:00-20:00 | 3000-4000K | 70% | 暖白光，放松过渡 |
| 20:00-06:00 | 2700-3000K | 40% | 暖黄光，促进睡眠 |

**特性：**
- 平滑过渡，避免色温突变
- 支持手动覆盖自动调节
- 可配置的时段和色温值

### 2. 运动预测 (Motion Prediction)

基于人员运动轨迹预测下一步移动方向：

**输入数据：**
- 位置坐标
- 运动方向 (角度)
- 运动速度
- 检测置信度

**预测能力：**
- 预测下一个目标区域
- 估算到达时间
- 运动趋势分析
- 预测准确率统计

### 3. 预测性照明控制 (Predictive Control)

结合昼夜节律和运动预测，实现智能照明控制：

**控制策略：**
- **预热阶段**：预测到达时间 < 10秒，开启30%亮度
- **完全开启**：预测到达时间 < 5秒，开启正常亮度
- **节能关闭**：无人后延迟5秒关闭

**优化机制：**
- 误预测检测和自适应调整
- 连续误预测时延长预热时间
- 能耗统计和节省计算

## 安装

```bash
# 从项目根目录
pip install -r requirements.txt
```

## 快速开始

### 基本使用

```python
from building_energy.lighting import (
    CircadianRhythm,
    MotionPredictor,
    PredictiveLightingController,
    MotionEvent,
    ZoneLayout,
    ZoneConfig
)
from datetime import datetime

# 1. 创建昼夜节律控制器
circadian = CircadianRhythm(
    latitude=39.9,   # 北京纬度
    longitude=116.4  # 北京经度
)

# 获取当前推荐的色温和亮度
color_temp, brightness = circadian.get_lighting_state()
print(f"当前色温: {color_temp}K, 亮度: {brightness:.0%}")

# 2. 创建区域布局
layout = ZoneLayout()
layout.add_zone(ZoneConfig(
    id="zone_1",
    name="走廊入口",
    center=(100, 200),
    radius=80,
    neighbors={"east": "zone_2"}
))
layout.add_zone(ZoneConfig(
    id="zone_2",
    name="走廊中段",
    center=(300, 200),
    radius=80,
    neighbors={"west": "zone_1", "east": "zone_3"}
))

# 3. 创建运动预测器
predictor = MotionPredictor(layout)

# 4. 创建预测性照明控制器
controller = PredictiveLightingController(
    circadian=circadian,
    predictor=predictor
)

# 设置灯光变化回调
def on_light_change(zone_id, is_active, brightness, color_temp):
    print(f"区域 {zone_id}: 激活={is_active}, 亮度={brightness:.0%}, 色温={color_temp}K")

controller.set_light_change_callback(on_light_change)

# 5. 处理运动检测事件
event = MotionEvent(
    timestamp=datetime.now(),
    zone_id="zone_1",
    position=(100, 200),
    direction=0.0,  # 向东
    speed=1.2,      # 1.2 m/s
    confidence=0.9,
    track_id="person_1"
)

controller.on_motion_detected(event)

# 启动控制器后台线程
controller.start()

# ... 运行中 ...

# 停止控制器
controller.stop()
```

### 与现有系统集成

```python
from corridor_light.zone_controller import ZoneLightController
from corridor_light.light_zones import LightConfig, LightZone

# 创建现有控制器
light_config = LightConfig()
light_config.add_zone(LightZone(
    id="zone_1",
    name="走廊入口",
    x=100, y=200, radius=80,
    forward_zones=["zone_2"],
    backward_zones=[]
))

zone_controller = ZoneLightController(light_config)

# 创建预测性控制器并集成
controller = PredictiveLightingController(
    circadian=circadian,
    predictor=predictor,
    zone_controller=zone_controller  # 集成现有控制器
)
```

## 配置

配置文件位于 `config.yaml`，包含以下部分：

### 昼夜节律配置

```yaml
circadian:
  latitude: 39.9
  longitude: 116.4
  schedule:
    morning:
      start: "06:00"
      end: "09:00"
      color_temperature: 6500
      brightness: 1.0
  transition_minutes: 30
```

### 运动预测配置

```yaml
motion_prediction:
  history_size: 50
  confidence_threshold: 0.6
  typical_walking_speed: 1.2
  pixels_per_meter: 100
```

### 预测性控制配置

```yaml
predictive_control:
  preheat_threshold: 10.0      # 预热提前时间(秒)
  activate_threshold: 5.0      # 完全开启提前时间(秒)
  preheat_brightness: 0.3      # 预热亮度
  normal_brightness: 0.9       # 正常亮度
  preheat_timeout: 15.0        # 预热超时
  light_off_delay: 5.0         # 关灯延迟
```

## API 参考

### CircadianRhythm

```python
class CircadianRhythm:
    def get_color_temperature(timestamp=None) -> int:
        """获取当前推荐色温 (K)"""
    
    def get_brightness(timestamp=None) -> float:
        """获取当前推荐亮度 (0-1)"""
    
    def get_lighting_state(timestamp=None) -> Tuple[int, float]:
        """获取当前照明状态 (色温, 亮度)"""
    
    def set_manual_override(color_temp=None, brightness=None, duration_minutes=60):
        """设置手动覆盖"""
    
    def clear_manual_override():
        """清除手动覆盖"""
```

### MotionPredictor

```python
class MotionPredictor:
    def update(event: MotionEvent):
        """更新运动事件"""
    
    def predict_next_zone(current_zone, track_id=None, direction=None) -> Optional[str]:
        """预测下一个区域"""
    
    def estimate_arrival_time(from_zone, to_zone, current_speed=None) -> float:
        """估算到达时间 (秒)"""
    
    def predict_destination(current_event, prediction_horizon=3) -> List[Tuple[str, float, float]]:
        """预测目的地及概率 [(区域ID, 置信度, 到达时间), ...]"""
    
    def get_prediction_accuracy() -> float:
        """获取预测准确率"""
```

### PredictiveLightingController

```python
class PredictiveLightingController:
    def on_motion_detected(event: MotionEvent):
        """处理运动检测事件"""
    
    def start():
        """启动控制器"""
    
    def stop():
        """停止控制器"""
    
    def get_stats() -> Dict:
        """获取统计信息"""
    
    def force_activate(zone_id: str):
        """强制激活区域"""
    
    def force_deactivate(zone_id: str):
        """强制关闭区域"""
```

## 测试

运行测试：

```bash
# 运行所有测试
python -m pytest building_energy/lighting/tests/

# 运行特定测试
python -m pytest building_energy/lighting/tests/test_circadian.py
python -m pytest building_energy/lighting/tests/test_motion_predictor.py

# 使用 unittest
python -m unittest building_energy.lighting.tests.test_circadian
python -m unittest building_energy.lighting.tests.test_motion_predictor
```

## 预期效果

### 人因照明
- ✅ 用户主观舒适度提升
- ✅ 符合昼夜节律，促进健康
- ✅ 平滑过渡，无感知调节

### 预测性开关
- ✅ 用户进入区域时灯已亮
- ✅ 减少等待感，提升体验
- ✅ 能耗增加 < 10%

## 代码审查清单

- [x] 色温调节符合昼夜节律
- [x] 运动预测准确率统计
- [x] 与corridor_light集成接口
- [x] 用户可手动覆盖自动调节
- [x] 代码包含类型注解
- [x] 关键函数有文档字符串
- [x] 包含单元测试

## 参考资源

- corridor_light系统: `corridor_light/` 目录
- 昼夜节律研究: https://doi.org/10.1016/j.jbiomech.2020.109914
- 人因照明标准: CIE S 026/E:2018

## 版本历史

- v1.0.0 (2026-03-22): 初始版本
  - 实现昼夜节律照明
  - 实现运动预测
  - 实现预测性照明控制
  - 与corridor_light系统集成

## 作者

智能照明控制专家团队

---

*文档版本: 1.0*  
*更新日期: 2026-03-22*
