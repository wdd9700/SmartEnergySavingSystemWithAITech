# PINN 训练环境 - 建筑热传导建模

物理信息神经网络 (Physics-Informed Neural Network, PINN) 模块，用于建筑温度分布预测。

## 功能特性

- **物理约束**: 基于热传导偏微分方程 (PDE) 训练神经网络
- **多后端支持**: PyTorch、MindSpore (华为昇腾)、PaddlePaddle (寒武纪)
- **NPU 推理**: 支持边缘设备上的实时推理 (<100ms)
- **ONNX 导出**: 标准格式导出，便于部署
- **CFD 数据生成**: 简化 CFD 模拟生成训练数据

## 快速开始

### 安装依赖

```bash
pip install deepxde>=1.15.0
pip install torch>=2.0.0  # 或 mindspore / paddlepaddle
pip install numpy matplotlib
pip install onnx onnxruntime
```

### 训练模型

```bash
# 完整训练
python building_energy/pinn/examples/train_example.py --epochs 10000

# 快速测试
python building_energy/pinn/examples/train_example.py --quick-test
```

### 运行推理

```bash
python building_energy/pinn/examples/inference_example.py \
    --model models/pinn/thermal_model.onnx \
    --outdoor-temp 30 \
    --solar-radiation 800 \
    --benchmark
```

## 核心组件

### 1. ThermalPINN 类

核心 PINN 模型，定义在 `thermal_pinn.py`。

```python
from building_energy.pinn import ThermalPINN, RoomParams, ExternalConditions

# 定义房间参数
room_params = RoomParams(
    wall_thickness=0.2,      # 墙厚 (m)
    room_volume=50.0,        # 房间体积 (m³)
    ac_position=(2.5, 2.5, 2.0),  # 空调位置
    ac_power=2.5,            # 空调功率 (kW)
    window_area=4.0,         # 窗户面积 (m²)
    insulation_factor=1.0    # 保温系数
)

# 创建模型
pinn = ThermalPINN(room_params, backend="pytorch")

# 设置几何和边界条件
pinn.setup_geometry(room_dimensions=(5.0, 5.0, 3.0))
pinn.setup_boundary_conditions(outdoor_temp=25.0)

# 设置训练数据
pinn.setup_data(n_domain=10000, n_boundary=2000, n_initial=1000)

# 构建和编译网络
pinn.build_network()
pinn.compile_model(learning_rate=0.001)

# 训练
history = pinn.train(epochs=10000)

# 预测
conditions = ExternalConditions(
    outdoor_temp=30.0,
    solar_radiation=800.0,
    wind_speed=5.0,
    time_of_day=14.0
)
temperatures = pinn.predict(conditions)
```

### 2. 数据生成器

使用简化 CFD 生成训练数据：

```python
from building_energy.pinn.data_generator import CFDDataGenerator, SimulationConfig

# 配置模拟
sim_config = SimulationConfig(
    nx=20, ny=20, nz=10,    # 网格分辨率
    dt=60.0,                # 时间步长 (秒)
    n_hours=24              # 模拟时长
)

# 创建生成器
generator = CFDDataGenerator(room_params, sim_config)

# 生成训练数据
data = generator.generate_training_data(n_samples=5000)

# 验证数据
is_valid = generator.validate_data(data)
```

### 3. 训练器

完整的训练流程：

```python
from building_energy.pinn.trainer import PINNTrainer, TrainingConfig

# 配置训练
config = TrainingConfig(
    epochs=10000,
    learning_rate=0.001,
    n_training_samples=5000,
    early_stopping=True,
    patience=1000
)

# 创建训练器
trainer = PINNTrainer(room_params, config, output_dir="models/pinn")

# 执行完整训练流程
trainer.generate_data()
trainer.setup_model(backend="pytorch")
trainer.train()
trainer.validate()
trainer.save_model()
```

### 4. ONNX 导出与 NPU 适配

```python
from building_energy.pinn.export import ONNXExporter, NPUAdapter

# 导出到 ONNX
exporter = ONNXExporter(input_shape=(1, 4))
exporter.export_from_pytorch(model, "models/pinn/thermal_model.onnx")

# NPU 推理
adapter = NPUAdapter(backend="ascend")  # 或 "mlu", "onnxruntime"
adapter.load_model("models/pinn/thermal_model.onnx")

# 预测
prediction = adapter.predict(input_data)

# 性能测试
benchmark = adapter.benchmark(n_iterations=100)
```

## 配置参数

配置文件位于 `config.yaml`：

```yaml
# 后端设置
backend:
  preferred: "pytorch"
  fallback_order: ["pytorch", "mindspore", "paddle"]

# 网络架构
network:
  layer_sizes: [4, 64, 64, 64, 64, 64, 64, 1]
  activation: "tanh"

# 训练参数
training:
  epochs: 10000
  learning_rate: 0.001
  batch_size: 256
  
# NPU 设置
npu:
  target: "onnxruntime"  # 或 "ascend", "mlu"
```

## 物理模型

### 热传导方程

PINN 求解的简化热传导方程：

```
∂T/∂t = α * ∇²T + Q_ac + Q_solar - Q_loss

其中:
- α: 热扩散系数 (与墙厚、材料相关)
- Q_ac: 空调热源/冷源
- Q_solar: 太阳辐射热增益
- Q_loss: 通过墙体的热损失
```

### 边界条件

- **墙体**: Robin 边界条件，考虑室外温度和保温系数
- **窗户**: 太阳辐射热增益
- **空调**: 点热源/冷源，随距离衰减

## 性能指标

### 推理延迟

| 后端 | 设备 | 平均延迟 | P95延迟 |
|------|------|----------|---------|
| ONNX Runtime | CPU | ~50ms | ~70ms |
| MindSpore | Ascend 310 | ~20ms | ~30ms |
| PaddlePaddle | MLU 270 | ~25ms | ~35ms |

### 模型精度

- MSE: < 0.5°C²
- MAE: < 0.5°C
- 满足实时控制需求 (<100ms)

## 目录结构

```
building_energy/pinn/
├── __init__.py              # 模块初始化
├── thermal_pinn.py          # 核心 PINN 模型
├── data_generator.py        # CFD 训练数据生成
├── trainer.py               # 训练脚本
├── export.py                # ONNX 导出与 NPU 适配
├── config.yaml              # 配置文件
└── examples/                # 示例脚本
    ├── train_example.py     # 训练示例
    └── inference_example.py # 推理示例
```

## 依赖项

### 必需
- Python >= 3.8
- NumPy
- DeepXDE >= 1.15.0

### 后端 (至少一个)
- PyTorch >= 2.0.0
- MindSpore (华为昇腾)
- PaddlePaddle (寒武纪)

### 可选
- ONNX / ONNX Runtime (导出和推理)
- Matplotlib (可视化)

## 故障排除

### DeepXDE 导入错误
```bash
pip install --upgrade deepxde>=1.15.0
```

### CUDA 内存不足
减小 batch size 或网络规模：
```python
network_config = {
    "layer_size": [4, 32, 32, 32, 1]  # 减小网络
}
```

### NPU 不可用
自动降级到 CPU/GPU：
```python
pinn = ThermalPINN(room_params, backend="auto")  # 自动选择
```

## 参考资料

- [DeepXDE 文档](https://deepxde.readthedocs.io/)
- [热传导方程](https://en.wikipedia.org/wiki/Heat_equation)
- [PINN 综述](https://maziarraissi.github.io/PINNs/)

## 许可证

MIT License
