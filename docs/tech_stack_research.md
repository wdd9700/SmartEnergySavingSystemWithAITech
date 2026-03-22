# 技术栈调研报告

**生成日期**: 2026年3月21日  
**调研范围**: PINN/CFD训练、车牌颜色识别、数字孪生可视化、YOLO路径分析

---

## 一、PINN/CFD训练框架 (建筑能源数字孪生)

### 1.1 推荐方案: DeepXDE

**项目信息**:
- **GitHub**: https://github.com/lululxvi/deepxde
- **PyPI**: `pip install DeepXDE`
- **版本**: 1.15.0 (2025年12月发布)
- **Stars**: 4k+
- **维护状态**: 活跃维护

**核心功能**:
- Physics-Informed Neural Network (PINN) 完整实现
- 支持前向/反向 ODE/PDE 求解
- 热传导方程示例内置 (heat.rst)
- 时间依赖PDE支持 (TimePDE)
- 边界条件: DirichletBC, NeumannBC, PeriodicBC, IC
- 梯度增强PINN (gPINN)
- 硬约束PINN (hPINN)

**后端支持**:
- TensorFlow 1.x / 2.x
- PyTorch
- JAX
- PaddlePaddle
- MindSpore

**建筑热传导应用示例**:
```python
import deepxde as dde
import numpy as np

# 定义几何 (房间一维简化模型)
geom = dde.geometry.Interval(0, L)  # 墙厚度方向

def pde(x, u):
    # 热传导方程: du/dt = alpha * d²u/dx²
    du_t = dde.grad.jacobian(u, x, i=0, j=1)  # 时间导数
    du_xx = dde.grad.hessian(u, x, i=0, j=0)  # 空间二阶导
    return du_t - alpha * du_xx

# 边界条件
bc = dde.icbc.DirichletBC(geom, lambda x: T_outside, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(geom, lambda x: T_initial, lambda _, on_initial: on_initial)

# 神经网络
net = dde.nn.FNN([2] + [32] * 3 + [1], "tanh", "Glorot uniform")

# 训练
model = dde.Model(data, net)
model.compile("adam", lr=0.001)
model.train(epochs=10000)
```

**替代方案对比**:

| 框架 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| **DeepXDE** | 功能完整、文档丰富、多后端 | 学习曲线较陡 | 复杂PDE、研究项目 |
| NVIDIA Modulus | GPU优化、工业级 | 依赖NVIDIA生态 | 大规模工业仿真 |
| PyTorch + functorch | 灵活、PyTorch原生 | 需手动实现PINN逻辑 | 定制化需求 |
| NeuroDiffEq | 轻量、易用 | 功能相对简单 | 简单ODE/PDE |

**NPU适配建议**:
- DeepXDE支持多种后端，可选择适配目标NPU的框架
- 华为昇腾NPU: 使用MindSpore后端
- 寒武纪MLU: 使用PaddlePaddle后端
- 训练完成后导出ONNX，使用NPU推理引擎

---

## 二、车牌颜色识别 (油电分类)

### 2.1 技术方案

**核心思路**: YOLO车牌检测 + HSV/RGB颜色空间分析

**中国车牌颜色特征**:

| 车辆类型 | 车牌颜色 | RGB特征 | HSV特征 |
|---------|---------|---------|---------|
| 燃油车 | 蓝色 | R低, G中, B高 | H: 200-240° (蓝) |
| 电动车 | 绿色渐变 | R低, G高, B中 | H: 120-180° (绿) |
| 新能源车 | 绿底黑字 | G显著高于R/B | S: 高, V: 中高 |

### 2.2 实现方案

**方案A: 传统CV颜色分析 (推荐)**
```python
import cv2
import numpy as np

def classify_plate_color(plate_img):
    """
    基于HSV颜色空间分类车牌
    无需神经网络，计算量极小
    """
    # 转换到HSV
    hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)
    
    # 提取ROI (去除文字区域，只保留底色)
    h, w = hsv.shape[:2]
    roi = hsv[int(h*0.2):int(h*0.8), int(w*0.1):int(w*0.9)]
    
    # 计算平均HSV
    mean_h = np.mean(roi[:, :, 0])  # Hue
    mean_s = np.mean(roi[:, :, 1])  # Saturation
    
    # 分类逻辑
    if 100 <= mean_h <= 140 and mean_s > 80:  # 绿色范围
        return "electric"
    elif 200 <= mean_h <= 260 and mean_s > 80:  # 蓝色范围
        return "fuel"
    else:
        return "unknown"
```

**方案B: RGB比值法**
```python
def classify_by_rgb_ratio(plate_img):
    """
    基于R/G/B比值分类
    """
    # 提取底色区域
    mean_b = np.mean(plate_img[:, :, 0])  # Blue通道
    mean_g = np.mean(plate_img[:, :, 1])  # Green通道
    mean_r = np.mean(plate_img[:, :, 2])  # Red通道
    
    # 电动车: G > B > R
    # 燃油车: B > G > R
    if mean_g > mean_b:
        return "electric"
    else:
        return "fuel"
```

### 2.3 集成到现有系统

**修改文件**: `traffic_energy/detection/vehicle_detector.py`

**扩展内容**:
1. 使用YOLO检测车牌位置 (添加车牌检测类别)
2. 裁剪车牌区域
3. 应用颜色分析算法
4. 将结果写入 `Detection` 数据类

---

## 三、数字孪生可视化技术栈

### 3.1 前端3D可视化方案

**推荐方案: Three.js**

**项目信息**:
- **GitHub**: https://github.com/mrdoob/three.js
- **Stars**: 111k+
- **版本**: r160+
- **维护状态**: 非常活跃

**核心特性**:
- WebGL/WebGPU渲染
- 轻量级、跨浏览器
- 丰富的几何体、材质、光照
- 支持导入GLTF/GLB模型
- 动画系统、粒子系统
- 后期处理效果

**建筑数字孪生应用场景**:

| 功能 | Three.js实现 | 说明 |
|-----|-------------|------|
| 3D房间建模 | BoxGeometry + Mesh | 基础墙体、房间结构 |
| 温度热力图 | ShaderMaterial | 基于温度数据着色 |
| 空调/风扇位置 | 3D模型导入 | GLTF格式设备模型 |
| 人员流动 | Points + Animation | 粒子表示人员 |
| 实时数据绑定 | React Three Fiber | 数据驱动渲染 |

**代码示例**:
```javascript
import * as THREE from 'three';

// 创建房间场景
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth/window.innerHeight, 0.1, 1000);

// 房间几何体
const roomGeometry = new THREE.BoxGeometry(10, 3, 8);
const roomMaterial = new THREE.MeshBasicMaterial({ 
    color: 0x00ff00, 
    wireframe: true,
    transparent: true,
    opacity: 0.3
});
const room = new THREE.Mesh(roomGeometry, roomMaterial);
scene.add(room);

// 空调位置标记
const acGeometry = new THREE.BoxGeometry(1, 0.3, 0.3);
const acMaterial = new THREE.MeshBasicMaterial({ color: 0x0000ff });
const ac = new THREE.Mesh(acGeometry, acMaterial);
ac.position.set(2, 2, 0);  // 空调位置
scene.add(ac);

// 温度热力图 (基于PINN预测数据)
function updateHeatmap(temperatureData) {
    // 根据温度数据更新材质颜色
    const colors = temperatureData.map(t => {
        // 温度到颜色的映射
        const normalized = (t - 18) / (30 - 18);
        return new THREE.Color().setHSL(0.7 - normalized * 0.7, 1, 0.5);
    });
    // 更新几何体顶点颜色...
}
```

### 3.2 替代方案对比

| 方案 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| **Three.js** | 轻量、文档丰富、社区大 | 需手动处理复杂场景 | 大多数Web 3D应用 |
| Babylon.js | 功能完整、游戏导向 | 包体积较大 | 复杂交互场景 |
| Cesium.js | GIS专业、大地形 | 学习曲线陡 | 地理空间数据 |
| Unity WebGL | 可视化编辑器 | 加载慢、文件大 | 高质量可视化 |
| Unreal Web | 顶级画质 | 硬件要求高 | 高端展示 |

### 3.3 与后端数据集成

**推荐架构**:
```
PINN预测模型 (Python/DeepXDE)
    ↓ WebSocket/HTTP
前端 Three.js 可视化
    ↓ 实时渲染
数字孪生界面 (类似自动驾驶HMI)
```

**数据流**:
1. 传感器数据 → 后端
2. PINN模型预测温度分布
3. 温度场数据 → 前端
4. Three.js热力图更新

---

## 四、YOLO路径分析与轨迹聚类

### 4.1 轨迹数据处理

**现有基础**: `traffic_energy/detection/vehicle_tracker.py`
- `TrajectoryPoint` 数据结构已存在
- 轨迹点包含: timestamp, bbox, center, velocity, speed

### 4.2 路径-时间图生成

**算法方案**:
```python
import numpy as np
from sklearn.cluster import DBSCAN

def generate_path_time_map(trajectories, camera_topology):
    """
    生成路径-时间图用于拥堵分析
    
    Args:
        trajectories: List[Track] 轨迹列表
        camera_topology: 摄像头拓扑关系
    
    Returns:
        path_time_data: 路径-时间矩阵
    """
    # 1. 轨迹聚类 (识别常见路径)
    path_clusters = cluster_trajectories(trajectories)
    
    # 2. 计算每条路径的通行时间分布
    path_time_data = {}
    for cluster_id, paths in path_clusters.items():
        travel_times = [calculate_travel_time(p) for p in paths]
        path_time_data[cluster_id] = {
            'mean_time': np.mean(travel_times),
            'std_time': np.std(travel_times),
            'count': len(paths),
            'path_geometry': extract_path_geometry(paths)
        }
    
    return path_time_data

def cluster_trajectories(trajectories):
    """
    使用DBSCAN聚类相似轨迹
    """
    # 提取轨迹特征向量 (起点、终点、路径形状)
    features = []
    for traj in trajectories:
        if len(traj.trajectory) >= 2:
            start = traj.trajectory[0].center
            end = traj.trajectory[-1].center
            # 简化特征: 起点、终点、路径长度
            feature = [start[0], start[1], end[0], end[1], len(traj.trajectory)]
            features.append(feature)
    
    # DBSCAN聚类
    clustering = DBSCAN(eps=50, min_samples=5).fit(features)
    
    # 分组
    clusters = {}
    for idx, label in enumerate(clustering.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(trajectories[idx])
    
    return clusters
```

### 4.3 拥堵原因LLM识别

**集成方案**:
```python
def analyze_congestion_cause(traffic_data, image_crops):
    """
    使用VLM识别拥堵原因
    
    Args:
        traffic_data: 流量统计数据
        image_crops: 拥堵区域图像裁剪
    """
    # 构建提示词
    prompt = f"""
    交通状况分析:
    - 车辆密度: {traffic_data['density']:.2f} 辆/米
    - 平均速度: {traffic_data['avg_speed']:.2f} km/h
    - 车辆类型分布: {traffic_data['vehicle_types']}
    
    请分析造成拥堵的可能原因。
    """
    
    # 调用VLM (如GPT-4V, Qwen-VL等)
    # response = vlm_client.analyze(images=image_crops, prompt=prompt)
    
    return {
        'cause': 'possible_accident',  # 事故/施工/信号灯/节假日等
        'confidence': 0.85,
        'description': '检测到车辆聚集，可能有事故'
    }
```

### 4.4 RAG写入供交警查询

**数据结构设计**:
```python
@dataclass
class CongestionEvent:
    event_id: str
    timestamp: datetime
    location: Tuple[float, float]  # GPS坐标
    camera_ids: List[str]  # 相关摄像头
    severity: str  # low/medium/high/critical
    cause: str  # LLM识别的原因
    cause_confidence: float
    vehicle_count: int
    avg_speed: float
    recommended_action: str  # 建议措施
    status: str  # active/resolved
```

---

## 五、技术栈选型总结

### 5.1 建筑能源方向

| 功能模块 | 推荐技术 | 备选方案 | 备注 |
|---------|---------|---------|------|
| PINN训练 | DeepXDE + PyTorch | NVIDIA Modulus | 支持NPU后端 |
| 热传导仿真 | DeepXDE TimePDE | 自研FDM | 简化CFD |
| 数字孪生可视化 | Three.js | Babylon.js | WebGL渲染 |
| 3D模型格式 | GLTF/GLB | OBJ | 轻量高效 |
| 实时通信 | WebSocket | MQTT | 低延迟 |

### 5.2 交通能源方向

| 功能模块 | 推荐技术 | 备选方案 | 备注 |
|---------|---------|---------|------|
| 车牌检测 | YOLOv8/v12 | 专用车牌检测器 | 已有基础 |
| 油电分类 | HSV颜色分析 | RGB比值 | 无需NN |
| 轨迹聚类 | DBSCAN | K-Means | sklearn内置 |
| 路径-时间图 | 自研算法 | - | 基于轨迹点 |
| 拥堵识别 | VLM (Qwen-VL) | GPT-4V | 多模态分析 |
| RAG存储 | YouTu GraphRAG | Microsoft GraphRAG | 用户指定 |

### 5.3 依赖安装清单

```bash
# PINN/CFD
pip install DeepXDE

# 轨迹聚类
pip install scikit-learn

# Three.js (前端)
npm install three @react-three/fiber

# VLM (可选)
pip install transformers torch
```

---

## 六、开发优先级建议

### 阶段一: 核心功能 (2-3周)
1. DeepXDE环境搭建 + 热传导PINN训练
2. 车牌HSV颜色分析实现
3. Three.js基础场景搭建

### 阶段二: 功能完善 (2-3周)
4. 轨迹聚类 + 路径-时间图
5. VLM拥堵识别集成
6. 数字孪生数据绑定

### 阶段三: 优化部署 (1-2周)
7. NPU适配优化
8. RAG系统集成
9. 前端性能优化

---

*报告结束*
