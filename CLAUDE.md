# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个 iLQR (Iterative Linear Quadratic Regulator) 求解器实现,用于车辆运动规划和控制。项目包含 Python 和 C++ 两种实现,C++ 实现通过 pybind11 提供 Python 绑定以获得更好的性能。

## 核心架构

### 1. 求解器层次结构

- **ILQR / FastILQR** (`cilqr/ilqr.py`, `cilqr/fast_ilqr.py`): 主求解器类,实现 iLQR 算法的核心逻辑
  - `linearized_initial_guess()`: 通过 LQR 生成初始猜测
  - `backward()`: 反向传播计算增益矩阵 K 和前馈项 k
  - `forward()`: 前向传播执行线搜索优化轨迹
  - `optimize()`: 主优化循环,包含增广拉格朗日法(ALM)处理约束

- **NewALILQR** (`cilqr/al_ilqr_cpp/new_al_ilqr.h`): C++ 模板实现,支持不同状态和控制维度的编译时优化

### 2. 节点抽象层

所有动力学节点继承自基类并实现以下接口:

- **ILQRNode** (`cilqr/ilqr_node.py`): Python 抽象基类
- **NewILQRNode** (`cilqr/al_ilqr_cpp/model/new_ilqr_node.h`): C++ 抽象基类

必须实现的方法:
- `dynamics(state, control)`: 动力学函数(通常使用 RK2 离散化)
- `dynamics_jacobian(state, control)`: 返回雅可比矩阵 A, B
- `cost()`: 目标函数(状态代价 + 控制代价 + 约束代价)
- `cost_jacobian()`: 目标函数的一阶导数
- `cost_hessian()`: 目标函数的二阶导数

### 3. 车辆模型

- **LatBicycleKinematicNode** (`cilqr/lat_bicycle_node.py`): 横向运动学自行车模型 (4维状态: x, y, θ, δ)
- **FullBicycleDynamicNode** (`cilqr/full_bicycle_dynamic_node.py`): 完整动力学自行车模型 (6维状态: x, y, θ, δ, v, a)
- **FastBicycleNode** (`cilqr/fast_bicycle_node.py`): 优化版本的自行车模型节点
- C++ 版本位于 `cilqr/al_ilqr_cpp/model/` 目录

### 4. 约束处理

使用增广拉格朗日法(Augmented Lagrangian Method)处理约束:

- **Constraints** (`cilqr/constraints.py`): 约束抽象基类
  - `constrains(x, u)`: 约束函数 c(x, u) ≤ 0
  - `projection()`: KKT 条件的投影函数
  - `augmented_lagrangian_cost()`: 增广拉格朗日代价
  - `update_lambda()`: 更新拉格朗日乘子
  - `update_mu()`: 更新惩罚因子

- **BoxConstraint** (`cilqr/box_constrains.py`): 盒式约束
- **LinearConstraints** (`cilqr/linear_constraints.py`): 线性约束
- **QuadraticConstraints** (`cilqr/al_ilqr_cpp/constraints/quadratic_constraints.h`): 二次约束(如障碍物)
- **DynamicLinearConstraints** (`cilqr/al_ilqr_cpp/constraints/dynamic_linear_constraints.h`): 动态线性约束

C++ 约束实现位于 `cilqr/al_ilqr_cpp/constraints/` 目录

## 常用命令

### Python 测试

```bash
# 测试横向自行车模型
cd cilqr
python test.py

# 测试完整动力学模型
python test_full.py
python test_fast_full.py

# 测试 pybind 绑定
python test_pybind.py
python test_lat_bicycle_pybind.py
python test_rectangle_obs_pybind.py
```

### C++ 构建和测试

使用 Bazel 构建系统:

```bash
cd cilqr/al_ilqr_cpp

# 构建所有目标
bazel build //...

# 运行测试
bazel run //:test_new_al_ilqr
bazel run //:test_lat_al_ilqr
bazel run //:test_dynamic_constraints_ilqr

# 构建 Python 绑定
bazel build //:ilqr_pybind
```

## 关键设计模式

### 1. 约束处理流程

```
初始化 λ=0, μ=1 → iLQR迭代 → 计算违反度 violation
  → 若 violation < 1e-3: 收敛
  → 若 1e-3 ≤ violation < 1e-1: 更新 λ
  → 若 violation ≥ 1e-1: 增大 μ (通常 μ *= 8)
```

### 2. 角度归一化

所有涉及角度的状态(θ, δ)必须调用 `normalize_angle()` 归一化到 (-π, π)

### 3. RK2 离散化

动力学积分使用二阶龙格-库塔方法:
```
k1 = f(x, u)
k2 = f(x + 0.5*dt*k1, u)
x_next = x + dt * k2
```

### 4. 模板特化 (C++)

C++ 实现使用编译时模板特化优化性能:
- 状态维度、控制维度、约束维度在编译时确定
- Eigen 库进行向量化加速
- 编译选项: `-O3 -march=native -DEIGEN_VECTORIZE`

## pybind11 绑定

Python 可以调用 C++ 实现的高性能求解器:

```python
from cilqr.al_ilqr_cpp.ilqr_pybind import (
    NewALILQR6_2,  # 6维状态, 2维控制
    NewBicycleNode6_2,
    BoxConstraints6_2
)
```

绑定定义位于:
- `cilqr/al_ilqr_cpp/ilqr_pybind.cc`: 主绑定文件
- `cilqr/al_ilqr_cpp/model/node_bind.h`: 节点绑定
- `cilqr/al_ilqr_cpp/constraints/constraints_bind.h`: 约束绑定

## 可视化工具

`viewer/` 目录包含可视化和调试工具(注意: 部分文件可能为空或不完整)

## 添加新的动力学模型

1. 继承 `ILQRNode` (Python) 或 `NewILQRNode` (C++)
2. 实现所有抽象方法(dynamics, jacobian, cost, hessian)
3. 在 dynamics_jacobian 中使用符号推导或自动微分
4. 测试: 使用数值微分验证雅可比和海森矩阵
5. 如果是 C++ 实现,在 pybind 文件中添加绑定

## 添加新的约束类型

1. 继承 `Constraints` 基类
2. 实现 `constrains()`, `constrains_jacobian()`, `constrains_hessian()`
3. 确保约束形式为 c(x, u) ≤ 0
4. 对于不等式约束,使用投影函数处理 KKT 条件
5. 在 C++ 中需要指定约束维度作为模板参数

## 注意事项

- 所有矩阵和向量使用 numpy.ndarray (Python) 或 Eigen (C++)
- 成本函数的 Hessian 必须正定,必要时添加正则化项
- 线搜索参数 alpha 从 1.0 开始,失败时减半
- 约束违反度使用无穷范数 `np.linalg.norm(dc, ord=np.inf)`
- C++ 代码使用 Bazel 构建,依赖 Eigen 和 pybind11
