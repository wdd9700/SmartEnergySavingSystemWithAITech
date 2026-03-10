# 智能灯光系统 - 基于亮度分析的自动校准方案

## 架构概述

```
┌─────────────────────────────────────────────────────────────────┐
│                     亮度分析自动校准系统 v4.0                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │   视频输入        │───▶│  亮度提取模块     │                   │
│  │  (摄像头/视频)    │    │ BrightnessExtractor│                  │
│  └──────────────────┘    └────────┬─────────┘                   │
│                                    │                             │
│                                    ▼                             │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │   灯光控制        │    │  分区域亮度统计   │                   │
│  │ (开关灯接口)      │    │  - 平均亮度       │                   │
│  └────────┬─────────┘    │  - 标准差         │                   │
│           │              │  - 最大/最小值    │                   │
│           ▼              └────────┬─────────┘                   │
│  ┌──────────────────┐             │                             │
│  │  开灯帧 ◀───────┼─────────────┘                             │
│  │  关灯帧 ◀───────┘                                            │
│  └────────┬─────────┘                                           │
│           ▼                                                      │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │ 亮度对比分析器   │───▶│  灯光贡献度计算   │                   │
│  │ LightComparator  │    │  (开灯-关灯差值)  │                   │
│  └──────────────────┘    └────────┬─────────┘                   │
│                                   │                              │
│                                   ▼                              │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │ 灯光中心估算     │    │  照明半径估算     │                   │
│  │ - 梯度加权法     │    │  (阈值百分比法)   │                   │
│  │ - 高斯拟合法     │    │                  │                   │
│  └──────────────────┘    └────────┬─────────┘                   │
│                                   │                              │
│                                   ▼                              │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │  等亮度线分析    │    │  灯区配置生成     │                   │
│  │  (Isophotes)     │───▶│  LightZone       │                   │
│  └──────────────────┘    └────────┬─────────┘                   │
│                                   │                              │
└───────────────────────────────────┼──────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    多摄像头校准系统                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                         │
│  │ Camera 1 │ │ Camera 2 │ │ Camera N │ ...                     │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘                         │
│       │            │            │                               │
│       └────────────┼────────────┘                               │
│                    ▼                                             │
│         ┌──────────────────────┐                                │
│         │  共享灯光检测匹配     │                                │
│         │  (同一灯在不同画面)   │                                │
│         └──────────┬───────────┘                                │
│                    │                                             │
│                    ▼                                             │
│         ┌──────────────────────┐                                │
│         │  相对位置计算         │                                │
│         │  - 偏移量 (dx, dy)    │                                │
│         │  - 缩放因子 (scale)   │                                │
│         └──────────┬───────────┘                                │
│                    │                                             │
│                    ▼                                             │
│         ┌──────────────────────┐                                │
│         │  全局坐标系生成       │                                │
│         │  (Unified Coord Sys) │                                │
│         └──────────┬───────────┘                                │
│                    │                                             │
│                    ▼                                             │
│         ┌──────────────────────┐                                │
│         │  位置映射服务         │                                │
│         │  (Camera A → Camera B)│                               │
│         └──────────────────────┘                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 核心算法

### 1. 灯光中心估算

```python
# 方法1: 亮度梯度加权法
def estimate_by_gradient(gray_image):
    # 选择高亮度区域 (前20%)
    threshold = percentile(gray_image, 80)
    mask = gray_image > threshold
    
    # 加权平均位置
    weights = gray_image[mask]
    x_center = average(x_indices[mask], weights=weights)
    y_center = average(y_indices[mask], weights=weights)
    
    return (x_center, y_center)
```

### 2. 照明半径估算

```python
def estimate_illumination_radius(contribution_map, threshold_percent=50):
    max_contribution = max(contribution_map)
    threshold = max_contribution * (threshold_percent / 100)
    
    # 找到超过阈值的区域
    affected_regions = [r for r in regions if r.contribution >= threshold]
    
    # 计算到中心的最远距离
    max_distance = max(distance(center, region.center) for region in affected_regions)
    
    return max_distance
```

### 3. 多摄像头相对位置计算

```python
# 使用两个共享灯光计算
def compute_transform(cam1_light1, cam1_light2, cam2_light1, cam2_light2):
    # 参考系中的距离
    ref_dist = distance(cam1_light1, cam1_light2)
    # 目标系中的距离  
    target_dist = distance(cam2_light1, cam2_light2)
    
    # 缩放因子
    scale = target_dist / ref_dist
    
    # 偏移量 (基于第一个灯光)
    offset = cam2_light1 - cam1_light1 * scale
    
    return scale, offset
```

## 使用流程

### 单摄像头单灯校准
```python
from corridor_light.auto_calibrator import LightAutoCalibrator

calibrator = LightAutoCalibrator()

result = calibrator.calibrate_single_camera_single_light(
    video_source=0,
    light_controller=lambda on: gpio_control(on),
    light_id="light_0",
    delay=2.0
)

# 结果:
# - pixel_position: (x, y) 灯光在画面中的像素位置
# - normalized_position: (0-1, 0-1) 归一化位置
# - estimated_radius: 照明半径(像素)
```

### 多灯校准
```python
light_controllers = {
    "light_0": lambda on: control_light(0, on),
    "light_1": lambda on: control_light(1, on),
    "light_2": lambda on: control_light(2, on),
}

config = calibrator.calibrate_single_camera_multiple_lights(
    video_source=0,
    light_controllers=light_controllers
)

# 自动建立灯光间的关联 (前后关系)
config.save_to_file("light_config.json")
```

### 多摄像头校准
```python
camera_configs = [
    {'id': 'cam1', 'source': 0, 'light_controller': lambda on: light1(on)},
    {'id': 'cam2', 'source': 1, 'light_controller': lambda on: light1(on)},
]

# 两个摄像头观察同一个灯
calibrator = calibrate_multi_camera(
    camera_configs,
    shared_light_id="light_0"
)

# 结果包含:
# - 各摄像头相对参考摄像头的偏移
# - 缩放因子
# - 全局坐标系
```

## 可视化输出

1. **亮度网格图** (`test_brightness_vis.jpg`)
   - 网格划分和亮度值
   - 等亮度线轮廓
   - 估算的灯中心位置

2. **贡献度热力图** (`test_contribution_vis.jpg`)
   - 颜色表示灯光对各区域的贡献
   - 蓝色=低贡献，红色=高贡献
   - 显示贡献值 (+XX)

3. **多摄像头校准图**
   - 各摄像头检测到的灯位置
   - 全局坐标映射
   - 相对位置信息

## 技术特点

| 特性 | 说明 |
|------|------|
| 无需人工测量 | 通过亮度分析自动确定位置 |
| 适应性强 | 可处理不同安装角度和高度 |
| 精度可调 | 网格密度、阈值参数可调 |
| 可视化支持 | 生成校准过程的可视化结果 |
| 多摄像头 | 支持摄像头间的位置关系校准 |
| 人工修正接口 | 提供可视化结果供人工校准参考 |
