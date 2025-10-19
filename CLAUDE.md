# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目简介

这是一个高性能的 **iLQR (Iterative Linear Quadratic Regulator)** 求解器,专为车辆运动规划和控制设计。项目提供了 Python 和 C++ 双实现,通过 pybind11 实现无缝集成,Python 用于快速原型开发,C++ 提供生产级性能(性能提升 30-50 倍)。

核心算法基于:
- **iLQR**: 迭代线性二次调节器
- **ALM**: 增广拉格朗日法处理不等式约束
- **车辆模型**: 支持横向运动学模型(4 维状态)和完整动力学模型(6 维状态)

---

## 常用命令

### Python 开发

```bash
# 进入核心代码目录
cd cilqr

# 运行基础测试 - 横向自行车模型
python test.py

# 运行完整动力学模型测试
python test_full.py

# 运行优化版本测试(使用约束对象)
python test_fast_full.py

# 测试 C++ Python 绑定 - 完整模型
python test_pybind.py

# 测试 C++ Python 绑定 - 横向模型
python test_lat_bicycle_pybind.py

# 测试障碍物避让功能
python test_rectangle_obs_pybind.py
```

### C++ 构建与测试

```bash
# 进入 C++ 目录
cd cilqr/al_ilqr_cpp

# 构建所有目标
bazel build //...

# 构建 Python 绑定模块
bazel build //:ilqr_pybind.so

# 运行 C++ 测试 - 完整动力学模型
bazel run //:test_new_al_ilqr

# 运行 C++ 测试 - 横向模型
bazel run //:test_lat_al_ilqr

# 运行动态约束测试
bazel run //:test_dynamic_constraints_ilqr

# 运行单次优化测试
bazel run //:test_new_al_ilqr_signal

# 运行所有测试
bazel test //...
```

### 设置 Python 绑定路径

生成的 Python 绑定位于 `cilqr/al_ilqr_cpp/bazel-bin/ilqr_pybind.so`,需要在代码中添加到 Python 路径:

```python
import sys
sys.path.append("./al_ilqr_cpp/bazel-bin")  # 从 cilqr 目录运行
# 或
sys.path.append("./cilqr/al_ilqr_cpp/bazel-bin")  # 从项目根目录运行
```

---

## 核心架构

### 层次结构

```
ILQR 求解器 (ilqr.py / fast_ilqr.py / new_al_ilqr.h)
    │
    ├─► ILQRNode 节点列表
    │     ├─ 状态 (state) 和控制 (control)
    │     ├─ 动力学函数 dynamics() 及雅可比
    │     ├─ 代价函数 cost() 及导数
    │     └─ 约束处理(通过 Constraints 对象)
    │
    └─► Constraints 约束对象(可选)
          ├─ BoxConstraint: 盒式约束
          ├─ LinearConstraints: 线性约束
          ├─ QuadraticConstraints: 二次约束(C++ only)
          └─ DynamicLinearConstraints: 时变线性约束(C++ only)
```

### 关键概念

**节点 (Node)**:
- 每个时间步对应一个节点,包含该时刻的状态、控制、目标、代价函数和约束
- 基类: `ILQRNode` (Python) / `NewILQRNode<StateDim, ControlDim>` (C++)
- 具体实现:
  - 横向模型: `LatBicycleKinematicNode` / `NewLatBicycleNode<4, 1>`
  - 完整模型: `FullBicycleDynamicNode` / `NewBicycleNode6_2`

**求解器 (Solver)**:
- 管理节点列表,执行优化循环
- 主要方法:
  - `linearized_initial_guess()`: 使用 LQR 生成初始轨迹
  - `backward()`: 反向传播计算增益
  - `forward()`: 前向传播更新轨迹
  - `optimize()`: 主优化循环

**约束处理**:
- Python 基础版本: 约束直接内置在节点中
- Python 优化版本: 通过 `Constraints` 对象传递给 `FastBicycleNode`
- C++ 版本: 约束作为模板参数传递给节点

---

## 项目结构

### Python 核心模块 (`cilqr/`)

**求解器**:
- `ilqr.py`: 基础 iLQR 求解器
- `fast_ilqr.py`: 优化版求解器,支持约束对象

**节点实现**:
- `ilqr_node.py`: 节点抽象基类
- `lat_bicycle_node.py`: 横向自行车运动学模型(4 维状态: x, y, θ, δ)
- `full_bicycle_dynamic_node.py`: 完整动力学模型(6 维状态: x, y, θ, δ, v, a)
- `fast_bicycle_node.py`: 优化版节点,支持 4/6 维模型切换

**约束模块**:
- `constraints.py`: 约束抽象基类
- `box_constrains.py`: 盒式约束(状态和控制的上下界)
- `linear_constraints.py`: 一般线性不等式约束

**工具函数**:
- `rk2.py`: RK2 积分器
- `jac.py`, `jac_lat_dynamic.py`, `jac_full_dynamic.py`: 雅可比计算工具

### C++ 核心模块 (`cilqr/al_ilqr_cpp/`)

**求解器**:
- `new_al_ilqr.h`: 模板化高性能 iLQR 求解器

**车辆模型** (`model/`):
- `new_ilqr_node.h`: 节点基类模板
- `new_bicycle_node.h`: 通用自行车模型(支持 4 维和 6 维)
- `new_lat_bicycle_node.h`: 横向专用模型
- `node_bind.h`: pybind11 绑定

**约束模块** (`constraints/`):
- `constraints.h`: 约束基类
- `box_constraints.h`: 盒式约束
- `linear_constraints.h`: 线性约束
- `quadratic_constraints.h`: 二次约束(用于障碍物避让)
- `dynamic_linear_constraints.h`: 时变线性约束
- `constraints_bind.h`: pybind11 绑定

**构建文件**:
- `BUILD`: Bazel 构建规则
- `MODULE.bazel`: Bazel 模块配置(新版)
- `WORKSPACE`: Bazel 工作空间配置(旧版,包含 Eigen 和 pybind11 依赖)
- `eigen.BUILD`: Eigen 库构建规则

---

## 算法流程

### 主优化循环

```
初始化: linearized_initial_guess()
  ├─ 使用 LQR 生成初始轨迹
  └─ 初始化 ALM 参数: λ=0, μ=1

外层循环 (ALM): max_outer_iter 次
  │
  ├─ 内层循环 (iLQR): max_inner_iter 次
  │   ├─ backward(): 反向传播计算增益 K, k
  │   ├─ forward(): 前向传播更新轨迹(线搜索)
  │   └─ 检查收敛: |ΔJ| < tol
  │
  ├─ 计算约束违反度: violation
  │
  └─ 更新 ALM 参数:
      ├─ if violation < 1e-3: 收敛
      ├─ elif violation < 1e-1: 更新 λ
      └─ else: 增大 μ (μ ← 8μ)
```

### 代价函数

**基础代价**:
```
J = Σ[(x - x_goal)ᵀQ(x - x_goal) + uᵀRu]
```

**增广拉格朗日代价** (处理约束 c(x,u) ≤ 0):
```
L_A = J + (1/(2μ))(||P(λ - μc)||² - ||λ||²)
其中 P(·) 是投影到非负域的算子
```

---

## 车辆模型说明

### 横向自行车运动学模型

**状态**: `x = [x, y, θ, δ]` (位置、航向角、前轮转角)
**控制**: `u = [δ̇]` (前轮转角速率)
**假设**: 纵向速度 v 恒定

**动力学方程**:
```
ẋ = v cos(θ)
ẏ = v sin(θ)
θ̇ = (v/L) tan(δ)
δ̇ = u
```

**适用场景**: 低速规划、停车场景、纵向速度已知

### 完整自行车动力学模型

**状态**: `x = [x, y, θ, δ, v, a]` (位置、航向角、前轮转角、速度、加速度)
**控制**: `u = [δ̇, j]` (前轮转角速率、加加速度 jerk)

**动力学方程**:
```
ẋ = v cos(θ)
ẏ = v sin(θ)
θ̇ = (v/L) tan(δ)
δ̇ = u[0]
v̇ = a
ȧ = u[1]
```

**适用场景**: 高速规划、需要加减速的场景、完整运动规划

---

## 编译配置

### C++ 编译优化选项

所有 C++ 目标使用以下优化标志(在 `BUILD` 文件中):
```python
copts = [
    "-O3",                  # 最高优化级别
    "-march=native",        # 针对本地 CPU 优化(启用 AVX/SSE 等)
    "-faligned-new",        # C++17 对齐内存分配
    "-DEIGEN_VECTORIZE"     # 启用 Eigen 向量化
]
```

### Eigen 依赖配置

Eigen 通过 HTTP archive 方式获取,配置在 `WORKSPACE` 文件中:
- 版本: 3.3.7
- 来源: GitHub 或阿波罗 CDN 镜像

如果编译失败,检查 `eigen.BUILD` 文件中的头文件路径配置。

### pybind11 配置

- pybind11: v2.13.6
- pybind11_bazel: b162c7c 提交
- Python 配置通过 `python_configure()` 自动检测

---

## 开发指南

### 添加新的车辆模型

1. **Python 实现**:
   - 继承 `ILQRNode` 基类
   - 实现 `dynamics()`, `dynamics_jacobian()`, `cost()` 等方法
   - 参考 `lat_bicycle_node.py` 或 `full_bicycle_dynamic_node.py`

2. **C++ 实现**:
   - 继承 `NewILQRNode<StateDim, ControlDim>` 模板类
   - 实现纯虚函数: `dynamics()`, `dynamics_diff()`, `running_cost_diff()` 等
   - 参考 `model/new_bicycle_node.h`

### 添加新的约束类型

1. **Python**:
   - 继承 `Constraints` 基类
   - 实现 `constrains()`, `constrains_jacobian()`, `constrains_hessian()`
   - 参考 `box_constrains.py` 或 `linear_constraints.py`

2. **C++**:
   - 继承 `Constraints<ConstraintDim>` 模板类
   - 实现约束计算和导数方法
   - 参考 `constraints/box_constraints.h`

### 验证雅可比计算

使用数值微分验证解析雅可比:
```python
from scipy.optimize import approx_fprime

# 验证动力学雅可比
def dynamics_func(x):
    return node.dynamics(x, u)

A_numerical = approx_fprime(x, dynamics_func, epsilon=1e-6)
A_analytical = node.dynamics_jacobian(x, u)[0]
error = np.linalg.norm(A_numerical - A_analytical)
assert error < 1e-4, f"雅可比误差过大: {error}"
```

---

## 性能优化建议

### 选择实现版本

| 场景 | 推荐实现 | 预期性能 |
|------|---------|---------|
| 算法研究、原型开发 | Python | 开发快速,调试方便 |
| 在线 MPC (10+ Hz) | C++ | 运行时间 < 20ms |
| 离线轨迹优化 | Python/C++ | 根据实时性要求选择 |

### 参数调优

**时间步长 dt**:
- 低速 (< 10 m/s): `dt = 0.1 - 0.2 s`
- 中速 (10-20 m/s): `dt = 0.05 - 0.1 s`
- 高速 (> 20 m/s): `dt = 0.02 - 0.05 s`

**迭代次数**:
- 实时应用: `max_inner_iter=5-10`, `max_outer_iter=5-10`
- 离线规划: `max_inner_iter=20-50`, `max_outer_iter=20`

**权重矩阵**:
- 状态权重 `Q`: 对角矩阵,关注的状态维度权重较大(如 y, θ)
- 控制权重 `R`: 较大的 R 使控制更平滑,较小的 R 提高跟踪精度

---

## 常见问题排查

### 优化不收敛

**症状**: 代价振荡或约束违反度不降低

**解决方案**:
1. 放松约束边界
2. 减小 μ 增长因子(从 8.0 改为 4.0)
3. 增大外层迭代次数
4. 改善初始猜测(使用更接近可行解的初始状态)
5. 调整权重矩阵(增大 Q,减小 R)

### 角度跳变

**症状**: 航向角 θ 或前轮转角 δ 在 ±π 处跳变

**解决方案**:
确保动力学函数中对角度进行归一化:
```python
def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi
```

### C++ 编译失败

**Eigen 头文件找不到**:
- 检查 `WORKSPACE` 文件中的 Eigen 配置
- 确认 `eigen.BUILD` 文件路径正确
- 验证网络连接(Eigen 从 GitHub 下载)

**pybind11 相关错误**:
- 确认 Python 版本 >= 3.7
- 检查 Python 开发头文件已安装: `sudo apt install python3-dev`

### Python 绑定导入失败

**找不到 .so 文件**:
```python
ImportError: ilqr_pybind.so: cannot open shared object file
```

**解决方案**:
1. 检查 `.so` 文件是否生成: `ls cilqr/al_ilqr_cpp/bazel-bin/ilqr_pybind.so`
2. 确认路径已添加到 `sys.path`
3. 验证文件权限: `chmod +x cilqr/al_ilqr_cpp/bazel-bin/ilqr_pybind.so`

---

## 项目维护

### Git 状态注意事项

- `bazel-*` 目录(bazel-bin, bazel-out 等)已被 `.gitignore` 忽略
- 不要提交编译生成的 `.so` 文件
- `MODULE.bazel.lock` 文件会随依赖更新而改变

### 文档

完整文档位于 `docs/` 目录:
- `00_项目概览.md`: 项目整体介绍
- `01_算法原理.md`: iLQR 和 ALM 数学原理
- `02_快速开始指南.md`: 安装和使用教程
- `03_API参考.md`: 完整 API 文档

### 许可证

项目采用 LICENSE 文件中规定的许可证。
