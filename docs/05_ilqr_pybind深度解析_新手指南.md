# ilqr_pybind 深度解析 - 新手完全指南

## 目录

1. [ilqr_pybind 模块概述](#1-ilqr_pybind-模块概述)
2. [C++ 类型系统与 pybind11 绑定机制](#2-c-类型系统与-pybind11-绑定机制)
3. [test_pybind.py 逐行代码解析](#3-test_pybindpy-逐行代码解析)
4. [完整技术流程图与调用链](#4-完整技术流程图与调用链)
5. [常见问题与使用建议](#5-常见问题与使用建议)

---

## 1. ilqr_pybind 模块概述

### 1.1 什么是 ilqr_pybind?

`ilqr_pybind` 是一个通过 **pybind11** 技术将 **C++ 实现的 iLQR 求解器**暴露给 Python 的桥接模块。它的核心价值在于:

- **性能提升**: C++ 实现比纯 Python 实现快 30-50 倍
- **无缝集成**: 在 Python 中可以像使用原生库一样使用 C++ 高性能代码
- **类型安全**: 保留了 C++ 的类型检查,同时支持 Python 的灵活性

### 1.2 模块架构图

```
┌─────────────────────────────────────────────────────────┐
│                   Python 层                              │
│  ┌─────────────────────────────────────────────────┐    │
│  │ test_pybind.py (用户代码)                        │    │
│  │  - 生成参考轨迹                                   │    │
│  │  - 配置约束和代价函数                             │    │
│  │  - 调用求解器                                     │    │
│  └──────────────────┬──────────────────────────────┘    │
│                     │ import ilqr_pybind                │
│  ┌──────────────────▼──────────────────────────────┐    │
│  │ ilqr_pybind.so (Python 模块)                     │    │
│  │  - BoxConstraints6_2                             │    │
│  │  - QuadraticConstraints6_2_5                     │    │
│  │  - NewBicycleNodeBoxConstraints6_2               │    │
│  │  - NewALILQR6_2                                  │    │
│  └──────────────────┬──────────────────────────────┘    │
└────────────────────┼──────────────────────────────────┘
                      │ pybind11 绑定
┌────────────────────▼──────────────────────────────────┐
│                   C++ 层                                │
│  ┌──────────────────────────────────────────────────┐  │
│  │ ilqr_pybind.cc (绑定代码)                          │  │
│  │  - PYBIND11_MODULE 宏                             │  │
│  │  - bind_* 模板函数                                │  │
│  └──────────────────┬──────────────────────────────┘  │
│                     │                                  │
│  ┌──────────────────▼──────────────────────────────┐  │
│  │ 核心 C++ 类库                                     │  │
│  │  ┌────────────────────────────────────────┐     │  │
│  │  │ NewALILQR<6,2>                         │     │  │
│  │  │  - optimize()                          │     │  │
│  │  │  - Backward(), Forward()               │     │  │
│  │  └────────────────────────────────────────┘     │  │
│  │  ┌────────────────────────────────────────┐     │  │
│  │  │ NewBicycleNode<ConstraintsType>        │     │  │
│  │  │  - dynamics() (车辆运动学)              │     │  │
│  │  │  - cost() (代价函数)                    │     │  │
│  │  └────────────────────────────────────────┘     │  │
│  │  ┌────────────────────────────────────────┐     │  │
│  │  │ Constraints 约束类                      │     │  │
│  │  │  - BoxConstraints (盒式约束)            │     │  │
│  │  │  - QuadraticConstraints (二次约束)      │     │  │
│  │  └────────────────────────────────────────┘     │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 1.3 关键概念速览

| 概念 | 说明 | 在代码中的体现 |
|------|------|----------------|
| **状态向量** | 描述车辆状态的 6 维向量 | `[x, y, θ, δ, v, a]` |
| **控制向量** | 描述控制输入的 2 维向量 | `[δ̇, j]` (转角速率、加加速度) |
| **约束** | 对状态和控制的限制条件 | `BoxConstraints`, `QuadraticConstraints` |
| **节点** | 时间序列上的单个离散点 | `NewBicycleNode` |
| **求解器** | 执行优化的核心算法 | `NewALILQR` |
| **增广拉格朗日法** | 处理约束优化的方法 | `update_lambda()`, `update_mu()` |

---

## 2. C++ 类型系统与 pybind11 绑定机制

### 2.1 C++ 模板参数化设计

#### 为什么使用模板?

C++ 使用模板来实现**编译时多态**,在编译阶段就确定类型,从而获得最高性能。以下是核心模板参数:

```cpp
// 核心模板参数
template <int state_dim, int control_dim>
class NewILQRNode { ... }

// 约束模板参数
template <int state_dim, int control_dim, int constraint_dim>
class Constraints { ... }
```

**模板参数说明**:

- `state_dim`: 状态向量维度 (4 或 6)
  - 4: 横向自行车模型 `[x, y, θ, δ]`
  - 6: 完整动力学模型 `[x, y, θ, δ, v, a]`
- `control_dim`: 控制向量维度 (1 或 2)
  - 1: 横向模型 `[δ̇]`
  - 2: 完整模型 `[δ̇, j]`
- `constraint_dim`: 约束数量 (可变)

#### 类型别名简化

为了提高代码可读性,使用了大量类型别名:

```cpp
// 在 NewILQRNode<6, 2> 中
using VectorState = Eigen::Matrix<double, 6, 1>;      // 状态向量
using VectorControl = Eigen::Matrix<double, 2, 1>;    // 控制向量
using MatrixA = Eigen::Matrix<double, 6, 6>;          // 状态雅可比
using MatrixB = Eigen::Matrix<double, 6, 2>;          // 控制雅可比
using MatrixQ = Eigen::Matrix<double, 6, 6>;          // 状态代价矩阵
using MatrixR = Eigen::Matrix<double, 2, 2>;          // 控制代价矩阵
```

### 2.2 pybind11 绑定命名规则

pybind11 通过特定的命名规则将 C++ 模板实例化为 Python 类。理解这个规则非常重要!

#### 命名规则

格式: `ClassName<type_params>` → `ClassNameParams` (Python 中)

**示例**:

| C++ 模板 | 模板参数 | Python 类名 |
|----------|----------|-------------|
| `BoxConstraints<6, 2>` | state_dim=6, control_dim=2 | `BoxConstraints6_2` |
| `QuadraticConstraints<6, 2, 5>` | state_dim=6, control_dim=2, constraint_dim=5 | `QuadraticConstraints6_2_5` |
| `NewBicycleNode<BoxConstraints<6,2>>` | ConstraintsType=BoxConstraints<6,2> | `NewBicycleNodeBoxConstraints6_2` |
| `NewALILQR<6, 2>` | state_dim=6, control_dim=2 | `NewALILQR6_2` |

#### 绑定代码示例

在 `ilqr_pybind.cc` 中:

```cpp
PYBIND11_MODULE(ilqr_pybind, m) {
    // 绑定盒式约束: 6 维状态, 2 维控制
    bind_box_constraints<6, 2>(m, "BoxConstraints6_2");

    // 绑定二次约束: 6 维状态, 2 维控制, 5 个约束
    bind_quadratic_constraints<6, 2, 5>(m, "QuadraticConstraints6_2_5");

    // 绑定自行车节点(带盒式约束)
    bind_new_bicycle_node<BoxConstraints<6, 2>>(m, "NewBicycleNodeBoxConstraints6_2");

    // 绑定 iLQR 求解器
    bind_new_al_ilqr<6, 2>(m, "NewALILQR6_2");
}
```

### 2.3 绑定函数模板详解

#### 约束绑定模板

```cpp
template <int state_dim, int control_dim>
void bind_box_constraints(py::module& m, const std::string& class_name) {
    constexpr int constraint_dim = 2 * (state_dim + control_dim);

    using BaseConstraintsType = LinearConstraints<state_dim, control_dim, constraint_dim>;
    using BoxConstraintsType = BoxConstraints<state_dim, control_dim>;

    py::class_<BoxConstraintsType, BaseConstraintsType>(m, class_name.c_str())
        .def(py::init<
            const Eigen::Matrix<double, state_dim, 1>&,      // state_min
            const Eigen::Matrix<double, state_dim, 1>&,      // state_max
            const Eigen::Matrix<double, control_dim, 1>&,    // control_min
            const Eigen::Matrix<double, control_dim, 1>&>(), // control_max
            py::arg("state_min"), py::arg("state_max"),
            py::arg("control_min"), py::arg("control_max")
        );
}
```

**要点**:
- `py::class_<T, Base>`: 声明 T 类继承自 Base
- `py::init<Args...>()`: 绑定构造函数
- `py::arg()`: 命名参数,使 Python 支持关键字参数

#### 节点绑定模板

```cpp
template <typename ConstraintsType>
void bind_new_bicycle_node(py::module& m, const std::string& class_name) {
    using NewBicycleNodeType = NewBicycleNode<ConstraintsType>;
    using BaseClass = NewILQRNode<6, 2>;
    using VectorState = Eigen::Matrix<double, 6, 1>;
    using MatrixQ = Eigen::Matrix<double, 6, 6>;
    using MatrixR = Eigen::Matrix<double, 2, 2>;

    py::class_<NewBicycleNodeType, BaseClass, std::shared_ptr<NewBicycleNodeType>>
        (m, class_name.c_str())
        .def(py::init<double, double, double,
                      const VectorState&,
                      const MatrixQ&,
                      const MatrixR&,
                      const ConstraintsType&>(),
             py::arg("L"),           // 车辆轴距
             py::arg("dt"),          // 时间步长
             py::arg("k"),           // 正则化系数
             py::arg("goal"),        // 参考状态
             py::arg("Q"),           // 状态代价矩阵
             py::arg("R"),           // 控制代价矩阵
             py::arg("constraints")  // 约束对象
        )
        .def("dynamics", &NewBicycleNodeType::dynamics)
        .def("cost", &NewBicycleNodeType::cost)
        // ... 更多方法
        ;
}
```

**要点**:
- `std::shared_ptr<T>`: 使用智能指针管理对象生命周期
- 绑定所有公共方法供 Python 调用

### 2.4 Eigen 矩阵与 NumPy 数组的自动转换

pybind11 提供了 Eigen 和 NumPy 之间的**自动转换**:

```cpp
#include <pybind11/eigen.h>  // 启用 Eigen 转换
```

**转换规则**:

| C++ 类型 | Python 类型 | 示例 |
|----------|-------------|------|
| `Eigen::Matrix<double, 6, 1>` | `numpy.ndarray` shape=(6,) | `np.array([0,0,0,0,10,0])` |
| `Eigen::Matrix<double, 6, 6>` | `numpy.ndarray` shape=(6,6) | `np.diag([1,1,1,1,1,1])` |
| `std::vector<Eigen::Matrix<...>>` | `list` of arrays | `[array1, array2, ...]` |

**这意味着**:
- Python 中传递 NumPy 数组会自动转换为 Eigen 矩阵
- C++ 返回的 Eigen 矩阵会自动转换为 NumPy 数组

---

## 3. test_pybind.py 逐行代码解析

### 3.1 导入模块与初始化

```python
import sys
import numpy as np
# 添加 C++ 编译生成的 Python 绑定模块路径
sys.path.append("/home/leo/workspace/repo/github/ilqr/cilqr/al_ilqr_cpp/bazel-bin")
import ilqr_pybind
```

**技术细节**:
- `ilqr_pybind` 是通过 Bazel 构建生成的 `.so` 文件(Linux)或 `.pyd` 文件(Windows)
- 必须将包含 `.so` 的目录添加到 `sys.path`
- 路径 `bazel-bin/` 是 Bazel 的默认输出目录

**生成的 .so 文件位置**:
```
cilqr/al_ilqr_cpp/bazel-bin/ilqr_pybind.so
```

**验证模块导入**:
```python
# 查看模块中可用的类和函数
print(dir(ilqr_pybind))
# 输出:
# ['BoxConstraints4_1', 'BoxConstraints6_2',
#  'NewALILQR4_1', 'NewALILQR6_2',
#  'NewBicycleNodeBoxConstraints6_2', ...]
```

### 3.2 生成参考轨迹 (第 36-90 行)

#### 函数签名与目的

```python
def generate_s_shape_goal_full(v, dt, num_points):
    """
    生成 S 形参考轨迹(完整动力学状态)

    参数:
        v: 期望速度 (m/s)
        dt: 时间步长 (s)
        num_points: 轨迹点数量

    返回:
        list: (num_points+1) 个状态向量 [x, y, theta, delta, v, a]
    """
```

#### 数学原理

**轨迹方程**:
```
x(t) = v * t                    # 匀速前进
y(t) = 50 * sin(0.1 * t)        # 正弦横摆
```

这个方程描述了车辆沿 x 轴匀速前进,同时在 y 方向做正弦运动,形成 S 形曲线。

**关键物理量计算**:

1. **一阶导数(速度方向)**:
```python
dx = v                          # x 方向速度恒定
dy = 50 * 0.1 * np.cos(0.1 * t) # y 方向速度
```

2. **二阶导数(加速度方向)**:
```python
ddx = 0                                 # x 方向加速度为 0
ddy = -50 * 0.1 * 0.1 * np.sin(0.1 * t) # y 方向加速度
```

3. **航向角(车辆朝向)**:
```python
theta = np.arctan2(dy, dx)
```
公式: θ = arctan2(dy/dt, dx/dt)

4. **曲率(路径弯曲程度)**:
```python
curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
```
公式: κ = (ẋÿ - ẏẍ) / (ẋ² + ẏ²)^(3/2)

5. **前轮转角(Ackermann 转向模型)**:
```python
delta = np.arctan(curvature * 1.0)  # 轴距 L = 1.0m
```
公式: δ = arctan(κ × L)

#### 代码执行示例

```python
v = 10          # 期望速度: 10 m/s
dt = 0.1        # 时间步长: 0.1 秒
num_points = 30 # 30 个点

goal_list_full = generate_s_shape_goal_full(v, dt, num_points)

# 第一个点 (t=0)
print(goal_list_full[0])
# 输出: [0, 0, 0, 0, 10, 0]
# 解释: 起点位于原点,航向向前,速度 10 m/s

# 第 10 个点 (t=1.0s)
print(goal_list_full[10])
# 输出: [10.0, 4.79, 0.048, 0.048, 10, 0]
# 解释: x 前进了 10m, y 偏移了 4.79m, 有轻微转角
```

### 3.3 配置优化器参数 - 盒式约束 (第 167-214 行)

#### 代价函数权重矩阵

```python
# 状态误差权重矩阵 Q (6×6 对角矩阵)
Q = np.diag([1e-1, 1e-1, 1e-0, 1e-9, 1e-6, 1e-6]) * 1e3
```

**展开后的实际值**:
```
Q = diag([100, 100, 1000, 0.001, 1, 1])
```

**物理意义**:
- `Q[0,0] = 100`: x 位置误差权重
- `Q[1,1] = 100`: y 位置误差权重
- `Q[2,2] = 1000`: **航向角误差权重**(最重要,误差代价最高)
- `Q[3,3] = 0.001`: 前轮转角误差权重(容忍度高)
- `Q[4,4] = 1`: 速度误差权重
- `Q[5,5] = 1`: 加速度误差权重

**设计原则**:
- 航向角权重最高,确保车辆朝向正确
- 位置误差次之,确保跟踪轨迹
- 转角权重最低,允许灵活调整

```python
# 控制输入权重矩阵 R (2×2 对角矩阵)
R = np.array([[1, 0], [0, 1]]) * 1e2
```

**展开后**:
```
R = diag([100, 100])
```

**物理意义**:
- `R[0,0] = 100`: 转角变化率 δ̇ 的权重(惩罚急转弯)
- `R[1,1] = 100`: 加速度变化率 j 的权重(惩罚急加减速)

**作用**: 权重越大,控制越平滑,但跟踪精度可能降低

#### 盒式约束定义

```python
# 状态约束
state_min = np.array([-1000, -1000, -2*np.pi, -10, -100, -10])
state_max = np.array([1000, 1000, 2*np.pi, 10, 100, 10])
```

**约束含义**:

| 状态 | 最小值 | 最大值 | 说明 |
|------|--------|--------|------|
| x | -1000 m | 1000 m | 位置范围(宽松) |
| y | -1000 m | 1000 m | 位置范围(宽松) |
| θ | -2π | 2π | 航向角范围(完整旋转) |
| δ | -10 rad | 10 rad | 前轮转角(非常宽松) |
| v | -100 m/s | 100 m/s | 速度范围 |
| a | -10 m/s² | 10 m/s² | 加速度范围 |

```python
# 控制约束
control_min = np.array([-0.2, -1])
control_max = np.array([0.2, 1])
```

**约束含义**:

| 控制 | 最小值 | 最大值 | 说明 |
|------|--------|--------|------|
| δ̇ | -0.2 rad/s | 0.2 rad/s | 转角变化率(限制打方向盘速度) |
| j | -1 m/s³ | 1 m/s³ | 加加速度(限制加速度变化率) |

#### 创建约束对象

```python
constraints = ilqr_pybind.BoxConstraints6_2(
    state_min, state_max, control_min, control_max
)
```

**背后发生的事情**:

1. **Python 端**: NumPy 数组被传递
2. **pybind11 转换层**: NumPy 数组自动转换为 `Eigen::Matrix<double, 6, 1>`
3. **C++ 端**: 调用 `BoxConstraints<6, 2>` 构造函数
4. **约束矩阵生成**: 生成内部的 A, B, C 矩阵

**内部表示** (线性约束形式 Ax + Bu ≤ C):

```cpp
// 对于 6 维状态 + 2 维控制,盒式约束有 16 个不等式:
// 6 个状态上界: x_i ≤ x_max_i  →  x_i ≤ x_max_i
// 6 个状态下界: x_i ≥ x_min_i  →  -x_i ≤ -x_min_i
// 2 个控制上界: u_i ≤ u_max_i  →  u_i ≤ u_max_i
// 2 个控制下界: u_i ≥ u_min_i  →  -u_i ≤ -u_min_i

// constraint_dim = 2 * (state_dim + control_dim) = 2 * (6 + 2) = 16
```

### 3.4 构建动力学节点列表 (第 216-235 行)

```python
ilqr_nodes_list = []
for i in range(horizon + 1):  # 31 个节点 (0 到 30)
    node = ilqr_pybind.NewBicycleNodeBoxConstraints6_2(
        L, dt, k, goal_list_full[i], Q, R, constraints
    )
    ilqr_nodes_list.append(node)
```

**每个节点的作用**:
- 代表时间序列上的一个离散点
- 包含该时刻的参考状态 `goal_list_full[i]`
- 封装了动力学模型、代价函数和约束

**参数解释**:

| 参数 | 值 | 说明 |
|------|------|------|
| `L` | 3.0 | 车辆轴距 (m) |
| `dt` | 0.1 | 时间步长 (s) |
| `k` | 0.001 | 正则化系数(用于数值稳定性) |
| `goal_list_full[i]` | 状态向量 | 该时刻的参考状态 |
| `Q` | 6×6 矩阵 | 状态代价权重 |
| `R` | 2×2 矩阵 | 控制代价权重 |
| `constraints` | 约束对象 | 盒式约束 |

**在 C++ 端发生的事情**:

```cpp
// 构造函数中
NewBicycleNode(double L, double dt, double k,
               const VectorState& goal,
               const MatrixQ& Q,
               const MatrixR& R,
               const ConstraintsType& constraints)
    : NewILQRNode<6, 2>(goal),
      constraints_(constraints),
      L_(L), dt_(dt), k_(k), Q_(Q), R_(R) {}
```

每个节点存储了:
- 车辆参数 (L, dt, k)
- 目标状态 (goal)
- 代价矩阵 (Q, R)
- 约束对象的副本

### 3.5 创建求解器并执行优化 (第 238-270 行)

#### 初始化求解器

```python
init_state = np.array([0, 0, 0, 0, v, 0])
#                      [x, y, θ, δ, v, a]
al_ilqr = ilqr_pybind.NewALILQR6_2(ilqr_nodes_list, init_state)
```

**求解器构造函数** (C++):

```cpp
NewALILQR(const std::vector<std::shared_ptr<NewILQRNode<6, 2>>>& ilqr_nodes,
          const VectorState& init_state)
    : ilqr_nodes_(ilqr_nodes), init_state_(init_state) {
    horizon_ = ilqr_nodes.size() - 1;  // 30
    x_list_.resize(6, 31);              // 状态轨迹
    u_list_.resize(2, 30);              // 控制序列
    // ... 初始化其他数据结构
}
```

**内部数据结构**:
- `x_list_`: 形状 (6, 31) - 存储优化后的状态轨迹
- `u_list_`: 形状 (2, 30) - 存储优化后的控制序列
- `K_list_`: 30 个反馈增益矩阵
- `k_list_`: 30 个前馈控制向量

#### 执行优化

```python
max_outer_iter = 50     # 外层迭代(增广拉格朗日法)
max_inner_iter = 100    # 内层迭代(iLQR)
max_violation = 1e-4    # 约束违反容忍度

al_ilqr.optimize(max_outer_iter, max_inner_iter, max_violation)
```

**优化流程** (在 C++ 的 `optimize()` 函数中):

```cpp
void optimize(int max_outer_iter, int max_inner_iter, double max_violation) {
    linearizedInitialGuess();  // 生成初始轨迹

    for (int iter = 0; iter < max_outer_iter; ++iter) {
        ILQRProcess(max_inner_iter, 1e-3);  // iLQR 内层优化

        double violation = ComputeConstraintViolation();

        if (violation < max_violation) {
            break;  // 收敛
        } else {
            if (violation > 5 * max_violation) {
                UpdateMu(100.0);  // 大幅增加惩罚系数
            } else {
                UpdateLambda();   // 更新拉格朗日乘子
            }
        }
    }
}
```

**详细步骤**:

1. **线性化初始猜测** (`linearizedInitialGuess`):
   - 使用 LQR 在参考轨迹附近生成初始轨迹
   - 初始化拉格朗日乘子 λ=0, 惩罚系数 μ=1

2. **外层循环** (增广拉格朗日法):
   - 调用内层 iLQR 优化
   - 计算约束违反度
   - 根据违反度更新 λ 和 μ

3. **内层循环** (`ILQRProcess`):
   ```cpp
   void ILQRProcess(int max_iter, double max_tol) {
       for (int iter = 0; iter < max_iter; ++iter) {
           CalcDerivatives();  // 计算雅可比和 Hessian
           Backward();         // 反向传播
           Forward();          // 前向传播(线搜索)

           if (cost_reduction < max_tol) break;
       }
   }
   ```

4. **反向传播** (`Backward`):
   - 从终点倒推到起点
   - 计算每个时刻的增益矩阵 K 和前馈项 k
   - 使用动态规划求解 Riccati 方程

5. **前向传播** (`Forward`):
   - 从起点推到终点
   - 使用 K 和 k 更新控制序列
   - 线搜索确定步长 α

#### 获取优化结果

```python
x_list = al_ilqr.get_x_list()  # shape: (6, 31)
u_list = al_ilqr.get_u_list()  # shape: (2, 30)

plot_x = x_list[0, :]  # x 坐标序列
plot_y = x_list[1, :]  # y 坐标序列
```

**返回值**:
- `x_list`: NumPy 数组,每列是一个时刻的状态
- `u_list`: NumPy 数组,每列是一个时刻的控制

### 3.6 添加障碍物约束 - 二次约束 (第 273-360 行)

#### 生成圆形障碍物约束

```python
def generate_cycle_equations(centre_x, centre_y, r, x_dims):
    """
    将圆形障碍物表示为二次不等式约束

    约束形式: x^T * Q * x + A^T * x + C <= 0
    圆形方程: (x - cx)^2 + (y - cy)^2 >= r^2
    """
```

**数学推导**:

原始圆形约束 (外部区域):
```
(x - cx)² + (y - cy)² ≥ r²
```

展开:
```
x² - 2cx·x + cx² + y² - 2cy·y + cy² ≥ r²
```

重新排列为标准形式 `x^T Q x + A^T x + C ≤ 0`:
```
-x² - y² + 2cx·x + 2cy·y + (cx² + cy² - r²) ≤ 0
```

**矩阵表示**:

```python
Q = [[−1, 0, 0, 0, 0, 0],      # -x² 项
     [0, −1, 0, 0, 0, 0],      # -y² 项
     [0, 0, 0, 0, 0, 0],
     ... (全零)]

A = [2·cx, 2·cy, 0, 0, 0, 0]  # 线性项

C = r² - cx² - cy²              # 常数项
```

**代码实现**:

```python
Q = np.zeros((6, 6))
A = np.zeros((1, 6))
C = np.zeros((1, 1))

# 常数项
C[0, 0] = r * r - centre_x * centre_x - centre_y * centre_y

# 二次项系数
Q[0, 0] = -1.0  # -x²
Q[1, 1] = -1.0  # -y²

# 线性项系数
A[0, 0] = 2 * centre_x  # 2cx·x
A[0, 1] = 2 * centre_y  # 2cy·y

return Q, A, C
```

#### 配置二次约束

```python
# 定义 5 个约束(1 个圆形 + 4 个线性)
Q_list = []
for i in range(5):
    Q_signal = np.zeros((6, 6))
    Q_list.append(Q_signal)

# 线性约束矩阵
A = np.zeros((5, 6))   # 状态系数
B = np.array([
    [0, 0],   # 第 1 个约束(将被圆形覆盖)
    [1, 0],   # u[0] ≤ -0.4
    [0, 1],   # u[1] ≤ -1
    [-1, 0],  # -u[0] ≤ -0.4
    [0, -1]   # -u[1] ≤ -1
])
C = np.array([
    [0], [-0.4], [-1], [-0.4], [-1]
])

# 设置圆形障碍物
circle_x = 30
circle_y = 11
circle_r = 6

Qc, Ac, Cc = generate_cycle_equations(circle_x, circle_y, circle_r, 6)

# 替换第 1 个约束为圆形障碍物
Q_list[0] = Qc
C[0, 0] = Cc.item()
A[0, :] = Ac
```

**约束数组结构** (Python 列表):

```python
Q_list = [
    Qc,               # 圆形障碍物的 Q 矩阵
    zeros(6,6),       # 线性约束(无二次项)
    zeros(6,6),
    zeros(6,6),
    zeros(6,6)
]
```

#### 创建二次约束对象

```python
quadratic_constraints = ilqr_pybind.QuadraticConstraints6_2_5(
    Q_list, A, B, C
)
```

**C++ 端转换**:

```cpp
// pybind11 自动转换
// Python list → std::array<Eigen::Matrix<double, 6, 6>, 5>
// Python ndarray → Eigen::Matrix<double, 5, 6>
```

#### 重新构建节点并优化

```python
quadratic_ilqr_nodes_list = []
for i in range(horizon + 1):
    node = ilqr_pybind.NewBicycleNodeQuadraticConstraints6_2_5(
        L, dt, k, goal_list_full[i], Q, R, quadratic_constraints
    )
    quadratic_ilqr_nodes_list.append(node)

q_al_ilqr = ilqr_pybind.NewALILQR6_2(quadratic_ilqr_nodes_list, init_state)
q_al_ilqr.optimize(max_outer_iter, max_inner_iter, max_violation)

q_x_list = q_al_ilqr.get_x_list()
q_u_list = q_al_ilqr.get_u_list()
```

**与盒式约束优化的区别**:
- 节点类型不同: `NewBicycleNodeQuadraticConstraints6_2_5`
- 约束对象不同: `QuadraticConstraints6_2_5`
- 优化结果会避开圆形障碍物

### 3.7 可视化对比 (第 363-417 行)

```python
plt.figure(figsize=(10, 6))
ax = plt.gca()

# 绘制圆形障碍物
circle = patches.Circle(
    (circle_x, circle_y), circle_r,
    edgecolor='green',
    facecolor='lightblue',
    alpha=0.5,
    linewidth=2,
    label='circle obstacle'
)
ax.add_patch(circle)

# 绘制三条轨迹
plt.plot(plot_x, plot_y, label='Optimized (Box)', c='b', marker='o')
plt.plot(goal_x, goal_y, label='Reference', c='r', marker='o')
plt.plot(q_plot_x, q_plot_y, label='Optimized (Obstacle)', c='g', marker='o')

# 标记起点和终点
plt.plot(init_state[0], init_state[1], 'ko', markersize=10, label='start')
plt.plot(goal_x[-1], goal_y[-1], 'k*', markersize=15, label='goal')

plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('iLQR Trajectory Optimization Comparison')
plt.legend()
plt.grid(True)
plt.show()
```

**可视化效果**:
- **红色虚线**: 理想参考轨迹(S 形曲线)
- **蓝色实线**: 仅考虑盒式约束的优化轨迹(可能穿过障碍物)
- **绿色实线**: 考虑障碍物的优化轨迹(绕开障碍物)
- **浅蓝圆形**: 需要避开的障碍物区域

---

## 4. 完整技术流程图与调用链

### 4.1 整体数据流

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Python 层: 准备数据                                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ generate_s_shape_goal_full()                         │   │
│  │  → 生成 31 个参考状态向量                             │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 配置代价矩阵 Q, R                                     │   │
│  │ 配置约束 state_min/max, control_min/max              │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Python → C++: 创建对象                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ constraints = ilqr_pybind.BoxConstraints6_2(...)     │   │
│  │   → pybind11 转换 NumPy → Eigen                      │   │
│  │   → 调用 C++ 构造函数                                 │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ for i in range(31):                                  │   │
│  │     node = NewBicycleNodeBoxConstraints6_2(          │   │
│  │         L, dt, k, goal[i], Q, R, constraints)        │   │
│  │     nodes_list.append(node)                          │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ al_ilqr = NewALILQR6_2(nodes_list, init_state)       │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. C++ 层: 优化求解                                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ al_ilqr.optimize(max_outer_iter, max_inner_iter, tol)│   │
│  └───────────────────────┬──────────────────────────────┘   │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ linearizedInitialGuess()                             │   │
│  │  - 使用 LQR 生成初始轨迹                              │   │
│  │  - 初始化 λ=0, μ=1                                   │   │
│  └──────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 外层循环 (增广拉格朗日法)                             │   │
│  │ ┌────────────────────────────────────────────────┐   │   │
│  │ │ ILQRProcess() ← 内层 iLQR 优化                 │   │   │
│  │ │ ┌──────────────────────────────────────────┐   │   │   │
│  │ │ │ CalcDerivatives()                        │   │   │   │
│  │ │ │  - dynamics_jacobian() → A, B           │   │   │   │
│  │ │ │  - cost_jacobian() → Jx, Ju             │   │   │   │
│  │ │ │  - cost_hessian() → Hx, Hu              │   │   │   │
│  │ │ └──────────────────────────────────────────┘   │   │   │
│  │ │ ┌──────────────────────────────────────────┐   │   │   │
│  │ │ │ Backward()                               │   │   │   │
│  │ │ │  - Riccati 递推                          │   │   │   │
│  │ │ │  - 计算增益 K, k                         │   │   │   │
│  │ │ └──────────────────────────────────────────┘   │   │   │
│  │ │ ┌──────────────────────────────────────────┐   │   │   │
│  │ │ │ Forward()                                │   │   │   │
│  │ │ │  - 线搜索确定步长 α                       │   │   │   │
│  │ │ │  - 更新轨迹 x_list, u_list               │   │   │   │
│  │ │ └──────────────────────────────────────────┘   │   │   │
│  │ └────────────────────────────────────────────────┘   │   │
│  │                                                          │   │
│  │ ┌────────────────────────────────────────────────┐   │   │
│  │ │ ComputeConstraintViolation()                   │   │   │
│  │ │  - 检查约束违反度                               │   │   │
│  │ └────────────────────────────────────────────────┘   │   │
│  │                                                          │   │
│  │ ┌────────────────────────────────────────────────┐   │   │
│  │ │ if violation < tol:                            │   │   │
│  │ │     break (收敛)                                │   │   │
│  │ │ else:                                          │   │   │
│  │ │     UpdateLambda() 或 UpdateMu()               │   │   │
│  │ └────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. C++ → Python: 返回结果                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ x_list = al_ilqr.get_x_list()                        │   │
│  │   → C++ Eigen::MatrixXd → NumPy ndarray             │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ u_list = al_ilqr.get_u_list()                        │   │
│  │   → C++ Eigen::MatrixXd → NumPy ndarray             │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. Python 层: 可视化                                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ matplotlib.pyplot.plot(x_list[0,:], x_list[1,:])     │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 关键函数调用链

#### 动力学计算调用链

```
Python: al_ilqr.optimize()
  ↓
C++: NewALILQR::optimize()
  ↓
C++: NewALILQR::ILQRProcess()
  ↓
C++: NewALILQR::CalcDerivatives()
  ↓
C++: ilqr_nodes_[i]->dynamics_jacobian(x, u)
  ↓
C++: NewBicycleNode::dynamics_jacobian()
  ↓ (内部调用)
C++: NewBicycleNode::dynamics_continuous()
  ↓ (RK2 积分)
返回: A (6×6), B (6×2) 雅可比矩阵
```

**dynamics_continuous 实现**:

```cpp
VectorState dynamics_continuous(const VectorState& state,
                                const VectorControl& control) const {
    VectorState x_dot;

    double theta = state(2);
    double delta = state(3);
    double v = state(4);
    double a = state(5);
    double u1 = control(0);  // δ̇
    double u2 = control(1);  // j

    // 自行车动力学方程
    x_dot(0) = v * cos(theta);                     // ẋ
    x_dot(1) = v * sin(theta);                     // ẏ
    x_dot(2) = v * tan(delta) / (L * (1 + k*v*v)); // θ̇
    x_dot(3) = u1;                                  // δ̇
    x_dot(4) = a;                                   // v̇
    x_dot(5) = u2;                                  // ȧ

    return x_dot;
}
```

#### 代价计算调用链

```
Python: al_ilqr.optimize()
  ↓
C++: NewALILQR::CalcDerivatives()
  ↓
C++: ilqr_nodes_[i]->cost(x, u)
  ↓
C++: NewBicycleNode::cost()
  ↓ (计算状态代价)
  state_cost = (x - goal)^T * Q * (x - goal)
  ↓ (计算控制代价)
  control_cost = u^T * R * u
  ↓ (计算约束代价)
C++: constraints_.augmented_lagrangian_cost(x, u)
  ↓
返回: total_cost = state_cost + control_cost + constraint_cost
```

**增广拉格朗日代价函数**:

```cpp
double augmented_lagrangian_cost(const VectorState& x,
                                  const VectorControl& u) {
    VectorConstraint c = constraints(x, u);  // 约束值
    double cost = 0.0;

    for (int i = 0; i < constraint_dim; ++i) {
        double proj = projection(lambda_[i] - mu_ * c[i]);
        cost += (proj * proj - lambda_[i] * lambda_[i]) / (2 * mu_);
    }

    return cost;
}

// 投影函数 (投影到非负域)
double projection(double x) {
    return (x > 0) ? x : 0;
}
```

**物理意义**:
- 当约束被满足 (c ≤ 0): 代价接近 0
- 当约束被违反 (c > 0): 代价快速增长
- λ 是拉格朗日乘子,μ 是惩罚系数

#### Backward 传播调用链

```
C++: NewALILQR::Backward()
  ↓ (初始化终端值函数)
  Vx = cost_jacobian_x[horizon]
  Vxx = cost_hessian_x[horizon]
  ↓ (从 t=horizon-1 倒推到 t=0)
  for t = horizon-1 down to 0:
    ↓ (获取雅可比)
    A = dynamics_jacobian_x[t]
    B = dynamics_jacobian_u[t]
    ↓ (计算 Q 函数)
    Qx = cost_jacobian_x[t] + A^T * Vx
    Qu = cost_jacobian_u[t] + B^T * Vx
    Qxx = cost_hessian_x[t] + A^T * Vxx * A
    Quu = cost_hessian_u[t] + B^T * Vxx * B
    Qux = B^T * Vxx * A
    ↓ (求解增益)
    K[t] = -Quu^(-1) * Qux
    k[t] = -Quu^(-1) * Qu
    ↓ (更新值函数)
    Vx = Qx + K^T * Quu * k + K^T * Qu + Qux^T * k
    Vxx = Qxx + K^T * Quu * K + K^T * Qux + Qux^T * K
  ↓
返回: K_list, k_list (反馈增益和前馈控制)
```

**Riccati 递推方程**:

值函数二次型:
```
V(x) = 1/2 * x^T * Vxx * x + Vx^T * x + V0
```

控制律:
```
u = K * (x - x_ref) + k
```

其中:
- K 是反馈增益 (控制 x 的偏差)
- k 是前馈项 (补偿模型误差)

#### Forward 传播调用链

```
C++: NewALILQR::Forward()
  ↓ (保存当前轨迹)
  pre_x_list = x_list
  pre_u_list = u_list
  old_cost = computeTotalCost()
  ↓ (线搜索)
  for alpha in [1.0, 0.5, 0.25, 0.125, ...]:
    ↓ (更新控制)
    for t = 0 to horizon-1:
      u[t] = u_pre[t] + K[t] * (x[t] - x_pre[t]) + alpha * k[t]
      ↓ (前向传播状态)
      x[t+1] = dynamics(x[t], u[t])
    ↓ (计算新代价)
    new_cost = computeTotalCost()
    ↓ (检查是否改善)
    if new_cost < old_cost:
      break (接受此步长)
  ↓
  if new_cost >= old_cost:
    ↓ (尝试并行线搜索)
    ParallelLinearSearch(alpha, best_alpha, best_cost)
    if best_cost < old_cost:
      UpdateTrajectoryAndCostList(best_alpha)
    else:
      ↓ (回退到原轨迹)
      x_list = pre_x_list
      u_list = pre_u_list
```

**并行线搜索优化**:

```cpp
void ParallelLinearSearch(double alpha, double& best_alpha, double& best_cost) {
    // 准备多个步长 [α, α/3, α/9, ..., α/3^(N-1)]
    // 使用 SIMD 并行计算 N 条轨迹的代价
    // 选择代价最小的步长
}
```

这利用了 Eigen 的向量化能力,一次性计算多个候选步长的代价。

### 4.3 内存管理与对象生命周期

#### Python 端

```python
# 创建约束对象
constraints = ilqr_pybind.BoxConstraints6_2(...)
# ↓ pybind11 在 C++ 堆上分配内存

# 创建节点列表
ilqr_nodes_list = []
for i in range(31):
    node = ilqr_pybind.NewBicycleNodeBoxConstraints6_2(...)
    ilqr_nodes_list.append(node)
# ↓ 每个 node 是一个 std::shared_ptr<NewBicycleNode>

# 创建求解器
al_ilqr = ilqr_pybind.NewALILQR6_2(ilqr_nodes_list, init_state)
# ↓ 求解器持有 ilqr_nodes 的 shared_ptr
```

**生命周期管理**:
1. Python 对象引用计数归零时,调用 C++ 析构函数
2. `std::shared_ptr` 确保节点对象在求解器销毁前不被释放
3. 约束对象被节点复制一份,各自独立

#### C++ 端

```cpp
// 求解器构造函数
NewALILQR(const std::vector<std::shared_ptr<NewILQRNode<6, 2>>>& ilqr_nodes,
          const VectorState& init_state)
    : ilqr_nodes_(ilqr_nodes),  // 拷贝 shared_ptr (引用计数+1)
      init_state_(init_state) {
    // 分配内部缓冲区
    x_list_.resize(6, 31);
    u_list_.resize(2, 30);
    // ...
}

// 析构函数 (自动生成)
~NewALILQR() {
    // shared_ptr 自动释放节点对象
    // Eigen 矩阵自动释放内存
}
```

---

## 5. 常见问题与使用建议

### 5.1 导入错误

#### 问题: `ModuleNotFoundError: No module named 'ilqr_pybind'`

**原因**:
1. `.so` 文件未生成
2. 路径未添加到 `sys.path`
3. Python 版本不兼容

**解决方案**:

```bash
# 1. 编译生成 .so 文件
cd cilqr/al_ilqr_cpp
bazel build //:ilqr_pybind.so

# 2. 检查文件是否存在
ls bazel-bin/ilqr_pybind.so

# 3. 在 Python 中添加路径
import sys
sys.path.append("./cilqr/al_ilqr_cpp/bazel-bin")
import ilqr_pybind

# 4. 检查 Python 版本
python --version  # 应该是 3.7+
```

#### 问题: `ImportError: undefined symbol`

**原因**: Eigen 或 pybind11 编译不匹配

**解决方案**:

```bash
# 清理并重新编译
bazel clean
bazel build //:ilqr_pybind.so

# 检查编译选项
# BUILD 文件中应包含:
copts = ["-O3", "-march=native", "-faligned-new", "-DEIGEN_VECTORIZE"]
```

### 5.2 类型错误

#### 问题: `TypeError: incompatible function arguments`

**常见原因**:

```python
# ❌ 错误: 使用 Python 列表
state_min = [-1000, -1000, -6.28, -10, -100, -10]
constraints = ilqr_pybind.BoxConstraints6_2(state_min, ...)
# TypeError: Expected numpy array

# ✅ 正确: 使用 NumPy 数组
state_min = np.array([-1000, -1000, -6.28, -10, -100, -10])
constraints = ilqr_pybind.BoxConstraints6_2(state_min, ...)
```

#### 问题: 矩阵维度不匹配

```python
# ❌ 错误: Q 矩阵维度错误
Q = np.diag([1, 1, 1, 1])  # 4×4 矩阵
node = ilqr_pybind.NewBicycleNodeBoxConstraints6_2(L, dt, k, goal, Q, R, constraints)
# Error: Expected 6×6 matrix

# ✅ 正确
Q = np.diag([1, 1, 1, 1, 1, 1])  # 6×6 矩阵
```

### 5.3 优化不收敛

#### 问题: 代价振荡,约束违反度不降低

**诊断**:

```python
# 添加调试输出 (需要在 C++ 端取消注释)
al_ilqr.optimize(max_outer_iter, max_inner_iter, max_violation)
# 输出会显示:
# inner_violation: 0.5
# inner_violation: 0.3
# inner_violation: 0.25  (持续下降)
```

**解决方案**:

1. **放松约束**:
```python
# 原始约束太严格
control_min = np.array([-0.1, -0.5])
control_max = np.array([0.1, 0.5])

# 放松约束
control_min = np.array([-0.3, -1.5])
control_max = np.array([0.3, 1.5])
```

2. **调整 μ 增长因子**:

在 `new_al_ilqr.h` 中修改:
```cpp
// 原始代码
if (inner_violation > 5 * max_violation) {
    UpdateMu(100.0);  // 增长过快
}

// 改为
if (inner_violation > 5 * max_violation) {
    UpdateMu(10.0);  // 缓慢增长
}
```

3. **增大迭代次数**:
```python
max_outer_iter = 100  # 增加外层迭代
max_inner_iter = 200  # 增加内层迭代
```

4. **改善初始猜测**:

```python
# 确保初始状态接近参考轨迹的起点
init_state = goal_list_full[0]  # 使用参考轨迹的起点
```

### 5.4 性能优化建议

#### 减少节点数量

```python
# 原始: 31 个节点
horizon = 30

# 优化: 减少到 15 个节点,但增大 dt
horizon = 15
dt = 0.2  # 时间步长加倍
```

**权衡**:
- 节点少 → 计算快,但精度降低
- 节点多 → 精度高,但计算慢

#### 调整编译优化级别

```python
# BUILD 文件中
copts = [
    "-O3",              # 最高优化级别
    "-march=native",    # 针对本地 CPU 优化 (启用 AVX/SSE)
    "-DEIGEN_VECTORIZE" # 启用 Eigen 向量化
]
```

**性能提升**:
- `-O3` vs `-O0`: 3-5 倍加速
- `-march=native`: 额外 20-30% 加速

#### 使用并行线搜索

在 C++ 代码中,`ParallelLinearSearch` 已默认启用,利用 SIMD 并行计算多个步长。

### 5.5 调试技巧

#### 打印中间结果

在 Python 端:
```python
# 检查约束对象
print(f"Constraint dimension: {constraints.get_constraint_dim()}")

# 检查节点列表
print(f"Number of nodes: {len(ilqr_nodes_list)}")

# 检查优化结果
x_list = al_ilqr.get_x_list()
print(f"Final state: {x_list[:, -1]}")
print(f"Goal state: {goal_list_full[-1]}")
print(f"Error: {np.linalg.norm(x_list[:, -1] - goal_list_full[-1])}")
```

#### 可视化约束违反度

```python
# 获取每个时刻的约束违反度 (需要在 C++ 端添加接口)
violations = []
for i in range(horizon + 1):
    x = x_list[:, i]
    u = u_list[:, i] if i < horizon else np.zeros(2)
    violation = ilqr_nodes_list[i].max_constraints_violation(x, u)
    violations.append(violation)

plt.plot(violations)
plt.xlabel('Time step')
plt.ylabel('Constraint violation')
plt.title('Constraint Violation Over Time')
plt.show()
```

#### 比较 Python 和 C++ 实现

```python
# 使用相同的参数运行 Python 版本和 C++ 版本
from cilqr.fast_ilqr import FastILQR  # Python 版本

# ... 配置相同的参数
python_solver = FastILQR(...)
python_solver.optimize()

cpp_solver = ilqr_pybind.NewALILQR6_2(...)
cpp_solver.optimize()

# 比较结果
python_x = python_solver.get_x_list()
cpp_x = cpp_solver.get_x_list()

print(f"Max difference: {np.max(np.abs(python_x - cpp_x))}")
```

### 5.6 扩展与定制

#### 添加自定义约束

如果要添加新的约束类型,需要:

1. **在 C++ 端定义新的约束类**:

```cpp
// constraints/my_custom_constraints.h
template <int state_dim, int control_dim, int constraint_dim>
class MyCustomConstraints : public Constraints<state_dim, control_dim, constraint_dim> {
public:
    // 实现约束计算
    VectorConstraint constraints(const VectorState& x, const VectorControl& u) const override {
        // 自定义约束逻辑
    }

    // 实现雅可比计算
    std::pair<MatrixCx, MatrixCu> constraints_jacobian(...) const override {
        // 自定义雅可比逻辑
    }

    // 实现 Hessian 计算
    std::tuple<...> constraints_hessian(...) const override {
        // 自定义 Hessian 逻辑
    }
};
```

2. **在绑定代码中暴露新类**:

```cpp
// constraints/constraints_bind.h
template <int state_dim, int control_dim, int constraint_dim>
void bind_my_custom_constraints(py::module& m, const std::string& class_name) {
    using MyConstraintsType = MyCustomConstraints<state_dim, control_dim, constraint_dim>;
    using BaseType = Constraints<state_dim, control_dim, constraint_dim>;

    py::class_<MyConstraintsType, BaseType>(m, class_name.c_str())
        .def(py::init</* 构造函数参数 */>())
        .def("constraints", &MyConstraintsType::constraints)
        // ... 其他方法
        ;
}

// ilqr_pybind.cc
PYBIND11_MODULE(ilqr_pybind, m) {
    // ...
    bind_my_custom_constraints<6, 2, 5>(m, "MyCustomConstraints6_2_5");
    // ...
}
```

3. **重新编译**:

```bash
bazel build //:ilqr_pybind.so
```

4. **在 Python 中使用**:

```python
import ilqr_pybind

my_constraints = ilqr_pybind.MyCustomConstraints6_2_5(...)
```

#### 修改车辆模型

如果要使用不同的车辆模型(例如 Ackermann 模型):

1. **创建新的节点类**:

```cpp
// model/my_vehicle_node.h
template <class ConstraintsType>
class MyVehicleNode : public NewILQRNode<6, 2> {
public:
    VectorState dynamics(...) const override {
        // 自定义动力学方程
    }

    std::pair<MatrixA, MatrixB> dynamics_jacobian(...) const override {
        // 自定义雅可比
    }

    // ... 其他方法
};
```

2. **绑定新节点**:

```cpp
template <typename ConstraintsType>
void bind_my_vehicle_node(py::module& m, const std::string& class_name) {
    using MyVehicleNodeType = MyVehicleNode<ConstraintsType>;
    using BaseClass = NewILQRNode<6, 2>;

    py::class_<MyVehicleNodeType, BaseClass, std::shared_ptr<MyVehicleNodeType>>
        (m, class_name.c_str())
        .def(py::init</* 参数 */>())
        // ... 方法
        ;
}
```

### 5.7 最佳实践总结

1. **参数调优顺序**:
   - 先调整代价矩阵 Q, R
   - 再放松约束边界
   - 最后调整迭代次数和容忍度

2. **代价矩阵设置原则**:
   - 航向角权重 > 位置权重 > 其他状态权重
   - 控制权重根据实际执行器能力设置

3. **约束设置原则**:
   - 初始约束设置宽松,逐步收紧
   - 确保初始状态满足约束

4. **性能优化顺序**:
   - 减少节点数量 (最有效)
   - 启用编译优化 (`-O3 -march=native`)
   - 减少迭代次数 (牺牲精度)

5. **调试流程**:
   - 先用 Python 版本验证算法正确性
   - 再用 C++ 版本提升性能
   - 对比两者结果确保一致性

---

## 附录: 术语表

| 术语 | 英文 | 说明 |
|------|------|------|
| **iLQR** | Iterative Linear Quadratic Regulator | 迭代线性二次调节器,一种优化控制算法 |
| **增广拉格朗日法** | Augmented Lagrangian Method (ALM) | 处理约束优化的方法 |
| **pybind11** | - | C++ 到 Python 的绑定库 |
| **Eigen** | - | C++ 线性代数库 |
| **雅可比矩阵** | Jacobian | 函数的一阶导数矩阵 |
| **Hessian 矩阵** | Hessian | 函数的二阶导数矩阵 |
| **Riccati 方程** | Riccati Equation | 最优控制中的递推方程 |
| **线搜索** | Line Search | 确定优化步长的方法 |
| **盒式约束** | Box Constraints | 变量上下界约束 |
| **二次约束** | Quadratic Constraints | 二次不等式约束 |
| **前馈控制** | Feedforward Control | 基于模型的开环控制 |
| **反馈控制** | Feedback Control | 基于状态偏差的闭环控制 |
| **曲率** | Curvature | 路径弯曲程度 |
| **Ackermann 转向** | Ackermann Steering | 车辆转向几何模型 |

---

## 参考资料

1. **项目文档**:
   - `CLAUDE.md`: 项目概览和常用命令
   - `docs/01_算法原理.md`: iLQR 和 ALM 数学原理

2. **外部资料**:
   - [pybind11 官方文档](https://pybind11.readthedocs.io/)
   - [Eigen 官方文档](https://eigen.tuxfamily.org/)
   - [iLQR 论文](https://homes.cs.washington.edu/~todorov/papers/TassaIROS12.pdf)

3. **相关代码**:
   - `cilqr/test_pybind.py`: 完整测试示例
   - `cilqr/al_ilqr_cpp/ilqr_pybind.cc`: 绑定代码
   - `cilqr/al_ilqr_cpp/new_al_ilqr.h`: 求解器实现

---

**文档版本**: v1.0
**最后更新**: 2025-10-19
**作者**: Claude Code (基于项目代码分析)
**适用项目版本**: commit c8dd28a
