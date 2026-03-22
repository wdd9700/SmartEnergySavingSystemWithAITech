# PINN 架构说明

## 概述

本文档详细说明建筑能源 PINN (Physics-Informed Neural Network) 模块的架构设计。

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         PINN Module                             │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Thermal    │  │    CFD       │  │    ONNX      │          │
│  │    PINN      │  │   Data Gen   │  │   Export     │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                   │
│         └─────────────────┼─────────────────┘                   │
│                           │                                     │
│                    ┌──────┴──────┐                             │
│                    │   Trainer   │                             │
│                    └──────┬──────┘                             │
│                           │                                     │
│                    ┌──────┴──────┐                             │
│                    │  NPU Adapter │                             │
│                    └─────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
```

## 核心组件详解

### 1. ThermalPINN (thermal_pinn.py)

#### 职责
- 定义物理信息神经网络架构
- 实现热传导 PDE
- 管理训练、预测和模型导出

#### 类结构
```python
class ThermalPINN:
    - room_params: RoomParams          # 房间物理参数
    - backend: str                     # 计算后端
    - network_config: dict             # 网络配置
    - model: dde.Model                 # DeepXDE 模型
    - alpha: float                     # 热扩散系数
    
    + __init__(room_params, backend)   # 初始化
    + define_pde(x, y)                 # 定义 PDE
    + setup_geometry()                 # 设置几何
    + setup_boundary_conditions()      # 设置边界条件
    + train(epochs)                    # 训练模型
    + predict(conditions)              # 预测温度
    + export_onnx(path)                # 导出 ONNX
```

#### PDE 定义

热传导方程实现：

```python
def define_pde(self, x, y):
    # 温度场
    T = y
    
    # 时间导数
    T_t = dde.grad.jacobian(y, x, i=0, j=3)
    
    # 空间导数 (Laplacian)
    T_xx = dde.grad.hessian(y, x, i=0, j=0)
    T_yy = dde.grad.hessian(y, x, i=1, j=1)
    T_zz = dde.grad.hessian(y, x, i=2, j=2)
    
    # 热源项
    Q_ac = self._compute_ac_effect(x)      # 空调
    Q_solar = self._compute_solar_effect(x) # 太阳辐射
    Q_loss = self._compute_heat_loss(x)     # 热损失
    
    # PDE 残差
    f = T_t - self.alpha * (T_xx + T_yy + T_zz) - Q_ac - Q_solar + Q_loss
    return f
```

### 2. CFDDataGenerator (data_generator.py)

#### 职责
- 生成合成训练数据
- 实现简化 CFD 模拟
- 验证数据物理合理性

#### 模拟流程

```
初始化温度场
    ↓
时间步进循环:
    ├── 应用边界条件
    ├── 计算 Laplacian
    ├── 时间积分
    └── 添加 AC 效应
    ↓
提取训练样本
```

#### 数值方法
- **空间离散**: 有限差分法 (FDM)
- **时间积分**: 显式欧拉法
- **稳定性条件**: CFL 条件

### 3. PINNTrainer (trainer.py)

#### 职责
- 协调数据生成和模型训练
- 管理训练流程和回调
- 保存和加载模型

#### 训练流程

```
┌─────────────┐
│  初始化配置  │
└──────┬──────┘
       ↓
┌─────────────┐
│ 生成训练数据 │ ← CFDDataGenerator
└──────┬──────┘
       ↓
┌─────────────┐
│  设置模型   │ ← ThermalPINN
└──────┬──────┘
       ↓
┌─────────────┐
│   训练循环   │ ← DeepXDE
└──────┬──────┘
       ↓
┌─────────────┐
│   验证评估   │
└──────┬──────┘
       ↓
┌─────────────┐
│  保存模型   │ → ONNX + Checkpoint
└─────────────┘
```

### 4. ONNXExporter (export.py)

#### 职责
- 导出模型到 ONNX 格式
- 优化 ONNX 模型
- 支持多后端导出

#### 导出流程

```
PyTorch Model → TorchScript → ONNX
                                    ↘
MindSpore Model → MindIR ───────────→ Optimized ONNX
                                    ↗
Paddle Model ───────────────────────→
```

### 5. NPUAdapter (export.py)

#### 职责
- 适配不同 NPU 后端
- 提供统一推理接口
- 性能基准测试

#### 支持的后端

| 后端 | 框架 | 设备 | 状态 |
|------|------|------|------|
| ONNX Runtime | onnxruntime | CPU/GPU | ✓ 已支持 |
| Ascend | MindSpore | 华为昇腾 | ✓ 已支持 |
| MLU | PaddlePaddle | 寒武纪 | ✓ 已支持 |

## 数据流

### 训练数据流

```
RoomParams ──┐
             ├──→ CFDDataGenerator ──→ Training Data ──→ ThermalPINN
Conditions ──┘                                              ↓
                                                      DeepXDE Training
                                                              ↓
                                                      Trained Model
```

### 推理数据流

```
ExternalConditions ──→ NPUAdapter ──→ ONNX Model ──→ Temperature Prediction
                              ↓
                    Ascend/MLU/CPU
```

## 网络架构

### PINN 网络结构

```
Input: [x, y, z, t] (4D)
    ↓
┌─────────────────────────────────────┐
│  Fully Connected Layer (4 → 64)    │
│  Tanh Activation                    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Fully Connected Layer (64 → 64)   │
│  Tanh Activation                    │
└─────────────────────────────────────┘
    ↓
         ... (4 more hidden layers)
    ↓
┌─────────────────────────────────────┐
│  Fully Connected Layer (64 → 1)    │
│  Linear Output                      │
└─────────────────────────────────────┘
    ↓
Output: Temperature T (1D)
```

### 损失函数

总损失 = PDE Loss + BC Loss + IC Loss + Data Loss

```python
loss = (
    w_pde * L_pde +      # PDE 残差
    w_bc * L_bc +        # 边界条件
    w_ic * L_ic +        # 初始条件
    w_data * L_data      # 数据拟合
)
```

## 物理模型

### 热传导方程

```
∂T/∂t = α∇²T + S

其中 S = Q_ac + Q_solar - Q_loss
```

### 热源项

#### 空调效应
```
Q_ac = P_ac * exp(-r/r₀) / V

其中:
- P_ac: 空调功率
- r: 距空调距离
- r₀: 影响半径
- V: 房间体积
```

#### 太阳辐射
```
Q_solar = G_solar * A_window * τ / V

其中:
- G_solar: 太阳辐射强度
- A_window: 窗户面积
- τ: 透射率
- V: 房间体积
```

#### 热损失
```
Q_loss = h_wall * (T - T_out) / d_wall

其中:
- h_wall: 墙体传热系数
- T_out: 室外温度
- d_wall: 墙厚
```

## 性能优化

### 训练优化

1. **自适应学习率**
   - Adam 优化器
   - 学习率衰减

2. **损失权重调整**
   - 动态调整 PDE/Data 权重
   - 基于训练进度

3. **早停机制**
   - 监控验证损失
   - 防止过拟合

### 推理优化

1. **模型量化**
   - FP32 → FP16/INT8
   - 减少内存占用

2. **批量推理**
   - 并行处理多个请求
   - 提高吞吐量

3. **缓存策略**
   - 缓存常用查询结果
   - 减少重复计算

## 扩展性设计

### 添加新的后端

```python
class NewBackendAdapter(NPUAdapter):
    def _load_model(self, path):
        # 实现模型加载
        pass
    
    def predict(self, input_data):
        # 实现推理
        pass
```

### 添加新的 PDE

```python
def define_custom_pde(self, x, y):
    # 定义新的物理方程
    pass
```

## 错误处理

### 降级策略

```
NPU 不可用 ──→ 尝试 GPU ──→ 尝试 CPU ──→ 物理近似
```

### 异常分类

| 异常类型 | 处理方式 |
|----------|----------|
| ImportError | 记录警告，使用备用后端 |
| RuntimeError | 重试或降级 |
| ValueError | 数据验证失败，跳过样本 |

## 测试策略

### 单元测试
- PDE 残差计算
- 数据生成器
- 网络前向传播

### 集成测试
- 端到端训练流程
- ONNX 导出/导入
- NPU 推理

### 性能测试
- 训练收敛性
- 推理延迟
- 内存使用

## 部署架构

### 边缘部署

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Sensor    │────→│  Edge Device │────→│   Actuator  │
│   Data      │     │  (NPU)       │     │   (AC)      │
└─────────────┘     └──────┬──────┘     └─────────────┘
                           │
                    ┌──────┴──────┐
                    │  PINN Model  │
                    │  (ONNX)      │
                    └─────────────┘
```

### 云端训练

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Training   │────→│   Cloud     │────→│   Model     │
│   Data      │     │   GPU       │     │   Registry  │
└─────────────┘     └─────────────┘     └─────────────┘
```

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| 1.0.0 | 2026-03-22 | 初始版本 |
