# linearizedInitialGuess 方法详细解析

## 目录
- [方法概述](#方法概述)
- [数学背景](#数学背景)
  - [LQR 问题定义](#lqr-问题定义)
  - [离散时间 Riccati 方程](#离散时间-riccati-方程)
- [算法原理](#算法原理)
  - [整体流程](#整体流程)
  - [反向传播 (Backward Pass)](#反向传播-backward-pass)
  - [前向传播 (Forward Pass)](#前向传播-forward-pass)
  - [参数重置](#参数重置)
- [代码实现详解](#代码实现详解)
  - [初始化阶段](#初始化阶段)
  - [Backward Pass 实现](#backward-pass-实现)
  - [Forward Pass 实现](#forward-pass-实现)
  - [增广拉格朗日参数重置](#增广拉格朗日参数重置)
- [实现细节与技巧](#实现细节与技巧)
  - [线性化点选择](#线性化点选择)
  - [正则化处理](#正则化处理)
  - [边界条件设置](#边界条件设置)
- [数学推导](#数学推导)
  - [动态规划方法](#动态规划方法)
  - [Riccati 方程推导](#riccati-方程推导)
  - [最优反馈增益计算](#最优反馈增益计算)
- [应用场景](#应用场景)
- [与标准 iLQR 的关系](#与标准-ilqr-的关系)
- [参考文献](#参考文献)

---

## 方法概述

`linearizedInitialGuess()` 是 iLQR 求解器的初始化方法，位于 `cilqr/al_ilqr_cpp/new_al_ilqr.h:648-696`。

**核心功能**: 使用离散时间 **LQR (Linear Quadratic Regulator)** 方法生成高质量的初始轨迹猜测，为后续的 iLQR 迭代提供良好的起点。

**为什么需要初始猜测？**
- iLQR 是一种局部优化方法，对初始轨迹敏感
- 随机初始化可能导致收敛缓慢或陷入局部最优
- LQR 生成的轨迹在目标状态附近已经接近最优，能大幅加快收敛速度

**算法流程概览**:
```
1. 初始化状态轨迹 (x_list_) 和控制序列 (u_list_)
2. Backward Pass: 通过 Riccati 方程反向传播计算 LQR 反馈增益 K
3. Forward Pass: 使用增益 K 前向仿真生成初始轨迹
4. 重置增广拉格朗日参数 (λ 和 μ)
```

---

## 数学背景

### LQR 问题定义

线性二次调节器 (LQR) 求解以下优化问题:

**连续时间形式**:
```
minimize  J = ∫₀ᵀ [(x(t) - x_goal)ᵀ Q (x(t) - x_goal) + u(t)ᵀ R u(t)] dt
          + (x(T) - x_goal)ᵀ Q_T (x(T) - x_goal)

subject to  ẋ(t) = A x(t) + B u(t)  (线性动力学)
```

**离散时间形式** (本实现采用):
```
minimize  J = Σₜ₌₀ᵀ⁻¹ [(xₜ - x_goal)ᵀ Q (xₜ - x_goal) + uₜᵀ R uₜ]
          + (x_T - x_goal)ᵀ Q_T (x_T - x_goal)

subject to  x_{t+1} = A xₜ + B uₜ  (离散化动力学)
```

**符号说明**:
- `xₜ ∈ ℝⁿ`: t 时刻的状态向量 (n = state_dim)
- `uₜ ∈ ℝᵐ`: t 时刻的控制输入 (m = control_dim)
- `x_goal`: 目标状态 (通常为期望轨迹)
- `Q ∈ ℝⁿˣⁿ`: 状态代价权重矩阵 (正半定)
- `R ∈ ℝᵐˣᵐ`: 控制代价权重矩阵 (正定)
- `A ∈ ℝⁿˣⁿ`: 状态转移矩阵 (动力学雅可比 ∂f/∂x)
- `B ∈ ℝⁿˣᵐ`: 控制输入矩阵 (动力学雅可比 ∂f/∂u)

### 离散时间 Riccati 方程

LQR 问题的最优解可以通过 **离散时间代数 Riccati 方程 (DARE)** 求解:

**Riccati 方程**:
```
Pₜ = Q + Aᵀ P_{t+1} (A - B Kₜ)
```

**反馈增益**:
```
Kₜ = (R + Bᵀ P_{t+1} B)⁻¹ (Bᵀ P_{t+1} A)
```

**最优控制律**:
```
uₜ* = -Kₜ (xₜ - x_goal)
```

**边界条件**:
```
P_T = Q_T  (终端代价的 Hessian)
```

**物理意义**:
- `Pₜ`: 从时刻 t 到终端 T 的 **cost-to-go** (剩余代价) 的二次近似系数
- `Kₜ`: 最优反馈增益矩阵，表示状态偏差与控制输入的线性关系
- 反向递推: 从终端 T 向初始时刻 0 计算，保证满足边界条件

---

## 算法原理

### 整体流程

`linearizedInitialGuess()` 方法采用 **两阶段** 策略:

```
┌─────────────────────────────────────────────────────────────┐
│ 阶段 1: Backward Pass (反向传播)                            │
│ ─────────────────────────────────────────────────────────   │
│  目标: 计算每个时间步的 LQR 反馈增益 Kₜ                     │
│                                                             │
│  for t = T-1 down to 0:                                    │
│    1. 获取动力学雅可比: A = ∂f/∂x, B = ∂f/∂u              │
│       (在目标状态 x_goal 和零控制处线性化)                  │
│    2. 计算反馈增益: Kₜ = (R + Bᵀ P B)⁻¹ (Bᵀ P A)          │
│    3. 更新 Riccati 矩阵: Pₜ = Q + Aᵀ P (A - B Kₜ)         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 阶段 2: Forward Pass (前向传播)                             │
│ ─────────────────────────────────────────────────────────   │
│  目标: 使用计算的增益 Kₜ 生成初始轨迹                       │
│                                                             │
│  x₀ = init_state (给定初始状态)                             │
│  for t = 0 to T-1:                                         │
│    1. 应用 LQR 控制律: uₜ = -Kₜ (xₜ - x_goal)              │
│    2. 前向仿真动力学: x_{t+1} = f(xₜ, uₜ)                  │
│       (使用实际的非线性动力学)                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 阶段 3: 参数重置                                            │
│ ─────────────────────────────────────────────────────────   │
│  重置所有节点的增广拉格朗日参数:                            │
│    - 拉格朗日乘子: λ = 0                                    │
│    - 惩罚系数: μ = 初始值 (通常为 1.0)                      │
└─────────────────────────────────────────────────────────────┘
```

### 反向传播 (Backward Pass)

**核心思想**: 使用 **动态规划** 从终端向后递推，计算每个时间步的最优反馈增益。

**详细步骤** (代码第 655-676 行):

1. **初始化边界条件** (第 657 行):
   ```cpp
   MatrixQ P = ilqr_nodes_[horizon_]->cost_hessian(zero_state_, zero_control_).first.Identity();
   ```
   - 理论: `P_T = Q_T` (终端代价的 Hessian)
   - 实现: 使用 `Identity()` 作为简化 (正则化技巧)

2. **反向循环** (第 660-676 行):
   ```cpp
   for (int t = horizon_ - 1; t >= 0; --t) {
       // 步骤 A: 获取线性化动力学
       auto dynamics_jacobian = ilqr_nodes_[t]->dynamics_jacobian(
           ilqr_nodes_[t]->goal(), VectorControl::Zero());
       MatrixA A = dynamics_jacobian.first;   // ∂f/∂x
       MatrixB B = dynamics_jacobian.second;  // ∂f/∂u

       // 步骤 B: 计算反馈增益 K
       MatrixK K = (ilqr_nodes_[t]->cost_hessian(...).second.Identity() * 20.0
                    + B.transpose() * P * B).inverse()
                   * (B.transpose() * P * A);
       K_list_[t] = K;

       // 步骤 C: 更新 Riccati 矩阵 P
       P = ilqr_nodes_[t]->cost_hessian(...).first.Identity()
           + A.transpose() * P * (A - B * K);
   }
   ```

**数学公式对应**:
```
步骤 B:  Kₜ = (R + Bᵀ P B)⁻¹ (Bᵀ P A)
         ↓
         K = (R_approx + Bᵀ P B)⁻¹ (Bᵀ P A)
         其中 R_approx = Identity * 20.0

步骤 C:  Pₜ = Q + Aᵀ P (A - B K)
         (直接按公式实现)
```

### 前向传播 (Forward Pass)

**核心思想**: 使用计算出的反馈增益 `Kₜ`，从初始状态开始前向仿真，生成初始轨迹。

**详细步骤** (代码第 678-688 行):

```cpp
for (int t = 0; t < horizon_; ++t) {
    VectorState goal_state = ilqr_nodes_[t]->goal();  // 第 t 时刻的目标状态
    MatrixK K = K_list_[t];                            // 第 t 时刻的反馈增益

    // 步骤 1: LQR 控制律
    u_list_.col(t) = -K * (x_list_.col(t) - goal_state);

    // 步骤 2: 前向仿真动力学 (使用实际非线性动力学!)
    x_list_.col(t + 1) = ilqr_nodes_[t]->dynamics(x_list_.col(t), u_list_.col(t));
}
```

**关键点**:
1. **控制律**: `uₜ = -Kₜ (xₜ - x_goal)`
   - 这是 LQR 的最优反馈控制律
   - 负号表示: 当状态偏离目标时，施加反向控制拉回

2. **动力学仿真**: 使用 `dynamics()` 而非线性化的 `A xₜ + B uₜ`
   - 虽然增益 K 是基于线性化计算的
   - 但轨迹生成使用实际的非线性动力学 (更准确)
   - 这是一种 **混合策略**: 线性控制器 + 非线性仿真

### 参数重置

**目的**: 清除之前优化的拉格朗日参数，为新一轮优化做准备。

**实现** (代码第 690-695 行):
```cpp
for (auto node : ilqr_nodes_) {
    node->reset_lambda();  // λ = 0
    node->reset_mu();      // μ = 初始值 (通常为 1.0)
}
```

**为什么需要重置？**
- 增广拉格朗日法中，λ 和 μ 是通过迭代更新的
- 每次调用 `optimize()` 时，应该从干净的状态开始
- 避免上次优化的残留参数影响新的优化过程

---

## 代码实现详解

### 初始化阶段

**代码位置**: 第 648-653 行

```cpp
template<int state_dim, int control_dim>
void NewALILQR<state_dim, control_dim>::linearizedInitialGuess() {
    // 初始化状态为给定的初始状态
    x_list_.col(0) = init_state_;

    // 初始化第一个控制为零 (可能未使用,因为会在后面覆盖)
    u_list_.col(0).setZero();
```

**说明**:
- `x_list_`: 状态轨迹矩阵，大小为 `state_dim × (horizon+1)`
  - 每一列是一个时间步的状态
  - `x_list_.col(0)` 是初始状态
- `u_list_`: 控制序列矩阵，大小为 `control_dim × horizon`
  - 每一列是一个时间步的控制
  - 初始化为零 (后面会被覆盖)

### Backward Pass 实现

**代码位置**: 第 655-676 行

```cpp
// ========== Backward Pass: 计算 LQR 反馈增益 ==========
// 终端代价的 Hessian 作为 Riccati 方程的边界条件 P_T = Q_T
MatrixQ P = ilqr_nodes_[horizon_]->cost_hessian(zero_state_, zero_control_).first.Identity();
```

**边界条件解析**:
- `ilqr_nodes_[horizon_]`: 终端节点 (时刻 T)
- `cost_hessian(zero_state_, zero_control_)`: 计算代价函数的 Hessian
  - 返回值: `std::pair<MatrixQ, MatrixR>` (状态 Hessian, 控制 Hessian)
  - `.first`: 取状态 Hessian
- `.Identity()`: 替换为单位矩阵 (简化处理)

**反向循环核心代码**:
```cpp
for (int t = horizon_ - 1; t >= 0; --t) {
    // 在目标状态和零控制处线性化动力学: x_{t+1} ≈ A x_t + B u_t
    auto dynamics_jacobian = ilqr_nodes_[t]->dynamics_jacobian(
        ilqr_nodes_[t]->goal(), VectorControl::Zero());
    MatrixA A = dynamics_jacobian.first;   // ∂f/∂x
    MatrixB B = dynamics_jacobian.second;  // ∂f/∂u
```

**线性化点选择**:
- `ilqr_nodes_[t]->goal()`: 目标状态 (期望轨迹的第 t 个点)
- `VectorControl::Zero()`: 零控制
- 假设: 在目标附近，零控制是合理的参考点

**反馈增益计算**:
```cpp
MatrixK K = (ilqr_nodes_[t]->cost_hessian(zero_state_, zero_control_).second.Identity() * 20.0
             + B.transpose() * P * B).inverse()
            * (B.transpose() * P * A);
K_list_[t] = K;
```

**公式分解**:
```
分子: Bᵀ P A
分母: R + Bᵀ P B
     ↓
     Identity * 20.0 + Bᵀ P B  (R 用 20*I 近似)

K = (20*I + Bᵀ P B)⁻¹ (Bᵀ P A)
```

**为什么用 `Identity * 20.0`？**
- 正则化: 避免 `R + Bᵀ P B` 接近奇异 (不可逆)
- 简化: 不依赖实际的代价 Hessian
- 经验值: 20.0 是经验调参的结果

**Riccati 矩阵更新**:
```cpp
P = ilqr_nodes_[t]->cost_hessian(zero_state_, zero_control_).first.Identity()
    + A.transpose() * P * (A - B * K);
```

**公式对应**:
```
Pₜ = Q + Aᵀ P_{t+1} (A - B K)
     ↓
P = I + Aᵀ P (A - B K)  (Q 用 I 近似)
```

### Forward Pass 实现

**代码位置**: 第 678-688 行

```cpp
// ========== Forward Pass: 使用 LQR 增益生成初始轨迹 ==========
for (int t = 0; t < horizon_; ++t) {
    VectorState goal_state = ilqr_nodes_[t]->goal();  // 第 t 时刻的目标状态
    MatrixK K = K_list_[t];                            // 第 t 时刻的反馈增益

    // LQR 控制律: u = -K (x - x_goal)
    u_list_.col(t) = -K * (x_list_.col(t) - goal_state);

    // 仿真前向动力学: x_{t+1} = f(x_t, u_t)
    x_list_.col(t + 1) = ilqr_nodes_[t]->dynamics(x_list_.col(t), u_list_.col(t));
}
```

**执行流程**:
1. **获取目标状态**: `goal_state = nodes[t]->goal()`
   - 每个节点存储了对应时刻的目标状态
   - 通常来自参考轨迹或期望路径

2. **应用控制律**: `u = -K * (x - x_goal)`
   - `x - x_goal`: 当前状态与目标状态的偏差
   - `K * (x - x_goal)`: 比例反馈控制
   - 负号: 产生反向控制力，拉回到目标

3. **前向仿真**: `x_{t+1} = f(x, u)`
   - 使用 `dynamics()` 函数 (实际的非线性动力学)
   - 例如自行车模型: 包含三角函数、非线性项
   - **不是** 使用线性化的 `A x + B u`

**为什么不用线性动力学？**
- 线性化仅在目标点附近准确
- 实际轨迹可能偏离目标较远
- 使用非线性动力学能生成更真实的轨迹

### 增广拉格朗日参数重置

**代码位置**: 第 690-695 行

```cpp
// ========== 重置增广拉格朗日参数 ==========
// 开始新的优化前,需要将所有节点的拉格朗日乘子和惩罚系数重置
for (auto node : ilqr_nodes_) {
    node->reset_lambda();  // λ = 0
    node->reset_mu();      // μ = 初始值 (通常为 1.0)
}
```

**参数说明**:
- `λ` (拉格朗日乘子): 对应约束的对偶变量
  - 初始化为 0
  - 在外层循环中通过 `UpdateLambda()` 更新

- `μ` (惩罚系数): 约束违反的惩罚强度
  - 初始化为 1.0
  - 在外层循环中通过 `UpdateMu()` 增加

---

## 实现细节与技巧

### 线性化点选择

**选择**: 在 `(x_goal, u=0)` 处线性化

**代码**:
```cpp
auto dynamics_jacobian = ilqr_nodes_[t]->dynamics_jacobian(
    ilqr_nodes_[t]->goal(),     // x = 目标状态
    VectorControl::Zero()        // u = 0
);
```

**优点**:
- 简单: 不需要迭代猜测轨迹
- 适用性: 对跟踪问题 (tracking problem) 效果好
- 稳定: 目标状态通常是系统的合理工作点

**局限性**:
- 假设目标状态是可达的
- 假设零控制在目标附近合理
- 对远离目标的初始状态可能不够准确

**改进方向**:
```cpp
// 可选: 在参考轨迹点处线性化
auto dynamics_jacobian = ilqr_nodes_[t]->dynamics_jacobian(
    reference_trajectory[t],        // 参考状态
    reference_control[t]            // 参考控制
);
```

### 正则化处理

**问题**: 直接使用代价 Hessian 可能导致数值问题

**解决方案**: 使用固定的正则化矩阵

**代码**:
```cpp
// R 的近似: 使用 20*I 替代实际的控制代价 Hessian
MatrixK K = (ilqr_nodes_[t]->cost_hessian(...).second.Identity() * 20.0
             + B.transpose() * P * B).inverse()
            * (B.transpose() * P * A);
```

**效果**:
1. **数值稳定性**:
   - 确保 `R + Bᵀ P B` 远离奇异 (行列式接近零)
   - 避免矩阵求逆失败

2. **正则化强度**:
   - 20.0: 经验值，平衡控制代价和状态跟踪
   - 增大: 更保守的控制 (控制输入更小)
   - 减小: 更激进的控制 (可能振荡)

3. **简化计算**:
   - 不需要计算实际的 Hessian
   - 统一的正则化参数，便于调试

**如何选择正则化系数？**
```
经验范围: 1.0 ~ 100.0
- 小值 (1.0): 适用于控制受限较少的系统
- 中值 (10.0 ~ 50.0): 通用选择
- 大值 (100.0): 适用于需要平滑控制的系统
```

### 边界条件设置

**理论边界条件**:
```
P_T = Q_T  (终端代价的 Hessian)
```

**实际实现**:
```cpp
MatrixQ P = ilqr_nodes_[horizon_]->cost_hessian(zero_state_, zero_control_).first.Identity();
```

**说明**:
- `cost_hessian(...).first`: 返回状态代价 Hessian
- `.Identity()`: 替换为单位矩阵

**为什么用 Identity？**
1. **简化**: 避免计算复杂的 Hessian
2. **稳定**: 单位矩阵是正定的，保证数值稳定
3. **经验**: 对大多数问题效果良好

**如果需要更精确的边界条件**:
```cpp
// 使用实际的终端代价 Hessian
MatrixQ P = ilqr_nodes_[horizon_]->cost_hessian(
    ilqr_nodes_[horizon_]->goal(),
    zero_control_
).first;
```

---

## 数学推导

### 动态规划方法

**目标**: 求解离散时间 LQR 问题的最优控制序列

**Bellman 最优性原理**:
```
V_t(x) = min_u [ L_t(x, u) + V_{t+1}(f(x, u)) ]
```

其中:
- `V_t(x)`: 从时刻 t 到终端 T 的最小代价 (值函数)
- `L_t(x, u)`: 时刻 t 的阶段代价
- `f(x, u)`: 系统动力学

**二次近似** (LQR 的关键假设):
```
V_t(x) ≈ V_t(x_goal) + (x - x_goal)ᵀ P_t (x - x_goal)
```

简化记号 (令 `δx = x - x_goal`, 假设 `V_t(x_goal) = 0`):
```
V_t(δx) = δxᵀ P_t δx
```

### Riccati 方程推导

**步骤 1: 展开 Bellman 方程**

阶段代价:
```
L_t(x, u) = (x - x_goal)ᵀ Q (x - x_goal) + uᵀ R u
          = δxᵀ Q δx + uᵀ R u
```

动力学约束 (线性化):
```
x_{t+1} = A x_t + B u_t
δx_{t+1} = A δx_t + B u  (因为 x_goal,t+1 = A x_goal,t)
```

值函数递推:
```
V_t(δx) = min_u [ δxᵀ Q δx + uᵀ R u + V_{t+1}(δx_{t+1}) ]
        = min_u [ δxᵀ Q δx + uᵀ R u + (A δx + B u)ᵀ P_{t+1} (A δx + B u) ]
```

**步骤 2: 对控制 u 求偏导**

展开二次项:
```
V_t(δx) = min_u [ δxᵀ Q δx + uᵀ R u + δxᵀ Aᵀ P_{t+1} A δx
                   + 2 uᵀ Bᵀ P_{t+1} A δx + uᵀ Bᵀ P_{t+1} B u ]
```

对 u 求导并令其为零:
```
∂V_t/∂u = 2 R u + 2 Bᵀ P_{t+1} A δx + 2 Bᵀ P_{t+1} B u = 0
```

求解最优控制:
```
(R + Bᵀ P_{t+1} B) u = -Bᵀ P_{t+1} A δx
u* = -(R + Bᵀ P_{t+1} B)⁻¹ Bᵀ P_{t+1} A δx
```

定义反馈增益:
```
K_t = (R + Bᵀ P_{t+1} B)⁻¹ Bᵀ P_{t+1} A

⟹ u* = -K_t δx
```

**步骤 3: 代入最优控制，求 P_t**

将 `u* = -K_t δx` 代入 Bellman 方程:
```
V_t(δx) = δxᵀ Q δx + δxᵀ K_tᵀ R K_t δx + (A δx - B K_t δx)ᵀ P_{t+1} (A δx - B K_t δx)
        = δxᵀ [ Q + K_tᵀ R K_t + (A - B K_t)ᵀ P_{t+1} (A - B K_t) ] δx
```

因为 `V_t(δx) = δxᵀ P_t δx`, 所以:
```
P_t = Q + K_tᵀ R K_t + (A - B K_t)ᵀ P_{t+1} (A - B K_t)
```

展开并化简 (使用 `K_t` 的定义):
```
P_t = Q + Aᵀ P_{t+1} (A - B K_t)
```

### 最优反馈增益计算

**公式总结**:

1. **反馈增益**:
   ```
   K_t = (R + Bᵀ P_{t+1} B)⁻¹ Bᵀ P_{t+1} A
   ```

2. **Riccati 矩阵**:
   ```
   P_t = Q + Aᵀ P_{t+1} (A - B K_t)
   ```

3. **边界条件**:
   ```
   P_T = Q_T
   ```

4. **最优控制律**:
   ```
   u_t* = -K_t (x_t - x_goal,t)
   ```

**计算流程**:
```
输入: A, B, Q, R, T (时域长度)

1. 初始化: P = Q_T
2. for t = T-1 down to 0:
     K_t ← (R + Bᵀ P B)⁻¹ Bᵀ P A
     P ← Q + Aᵀ P (A - B K_t)
3. 输出: K_0, K_1, ..., K_{T-1}
```

---

## 应用场景

### 1. 轨迹跟踪 (Trajectory Tracking)

**场景**: 车辆跟踪预定义的参考轨迹

**为什么适合 LQR 初始化？**
- 目标轨迹已知且连续
- 在目标附近的线性化是合理的
- LQR 能快速生成接近参考的初始轨迹

**示例**:
```cpp
// 设置参考轨迹
for (int t = 0; t <= horizon; ++t) {
    nodes[t]->set_goal(reference_trajectory[t]);
}

// 使用 LQR 初始化
solver->linearizedInitialGuess();

// 进行 iLQR 优化
solver->optimize(max_outer_iter, max_inner_iter, max_violation);
```

### 2. 稳定控制 (Stabilization)

**场景**: 将系统从当前状态稳定到平衡点

**示例**: 倒立摆稳定到垂直向上位置

**LQR 的优势**:
- 在平衡点附近的线性化非常准确
- LQR 控制律保证局部稳定性
- 初始轨迹已经具有稳定特性

### 3. 障碍物避障

**场景**: 在有障碍物的环境中规划轨迹

**使用方法**:
1. 先用 RRT/A* 等方法生成无碰撞的粗糙路径
2. 将路径设为 `goal` (目标轨迹)
3. 使用 `linearizedInitialGuess()` 生成初始猜测
4. iLQR 优化时添加障碍物约束，进一步细化轨迹

**优势**:
- LQR 初始化提供了平滑的基础轨迹
- 避免 iLQR 从零开始搜索 (可能卡在局部最优)

### 4. 模型预测控制 (MPC)

**场景**: 实时滚动优化

**流程**:
```
t = 0:
  1. 使用 LQR 初始化
  2. iLQR 优化
  3. 执行第一步控制 u_0

t = 1:
  1. 将上次的轨迹平移作为 warm start
  2. 或者重新用 LQR 初始化
  3. iLQR 优化
  4. 执行第一步控制 u_0
```

**何时重新初始化？**
- 如果参考轨迹变化大: 使用 LQR 重新初始化
- 如果只是小幅调整: 使用上次轨迹作为 warm start

---

## 与标准 iLQR 的关系

### iLQR 算法框架

标准 iLQR 算法包含两个主要阶段:

**1. Backward Pass (本文方法类似)**:
   - 线性化动力学: `x_{t+1} ≈ f(x̄_t, ū_t) + A δx + B δu`
   - 二次近似代价: `L(x, u) ≈ L(x̄, ū) + L_x δx + L_u δu + (1/2) δxᵀ L_xx δx + ...`
   - 计算反馈增益 `K` 和前馈项 `k`

**2. Forward Pass (本文方法简化)**:
   - 线搜索更新轨迹: `u = ū + K δx + α k`
   - 标准 iLQR 还包括线搜索步长 α 的调整

### linearizedInitialGuess 的特殊性

| 特性               | linearizedInitialGuess | 标准 iLQR Backward Pass |
|--------------------|------------------------|-------------------------|
| **线性化点**       | 目标状态 + 零控制      | 当前轨迹 (x̄, ū)        |
| **代价近似**       | 简化 (Identity 矩阵)   | 完整的二阶泰勒展开      |
| **前馈项 k**       | 无 (仅计算 K)          | 有 (k = -Q_uu⁻¹ Q_u)    |
| **值函数更新**     | 简化的 Riccati 方程    | 完整的动态规划更新      |
| **适用阶段**       | 初始化                 | 每次迭代                |

### 为什么不直接用标准 iLQR？

**linearizedInitialGuess 的优势**:
1. **计算速度快**: 只需一次反向传播，不需要迭代
2. **鲁棒性强**: 不依赖当前轨迹质量
3. **理论保证**: LQR 在线性系统下是全局最优的

**何时使用标准 iLQR**:
- 初始化之后的所有迭代
- 系统非线性较强时
- 需要处理约束时

### 完整的优化流程

```
┌────────────────────────────────────────────────────────────┐
│ 1. 初始化 (linearizedInitialGuess)                        │
│    ↓                                                       │
│    使用 LQR 方法生成初始轨迹 (x_list, u_list)             │
└────────────────────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────────────┐
│ 2. iLQR 迭代 (ILQRProcess)                                │
│                                                            │
│    for iter = 1 to max_iter:                              │
│      a) UpdateConstraints(): 更新动态约束                 │
│      b) CalcDerivatives(): 计算导数 (在当前轨迹处)        │
│      c) Backward(): 标准 iLQR 反向传播                    │
│         - 计算 K 和 k (包括前馈项)                        │
│         - 使用完整的代价和动力学导数                      │
│      d) Forward(): 线搜索前向传播                         │
│         - u_new = u + K δx + α k                          │
│         - 自动调整步长 α                                  │
└────────────────────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────────────┐
│ 3. 外层迭代 (增广拉格朗日法)                              │
│                                                            │
│    for outer_iter = 1 to max_outer_iter:                  │
│      a) 执行 ILQRProcess                                  │
│      b) 检查约束违反量                                    │
│      c) 更新 λ 或 μ                                       │
└────────────────────────────────────────────────────────────┘
```

---

## 参考文献

### 核心论文

1. **iLQR 算法**:
   - Li, W., & Todorov, E. (2004). *Iterative Linear Quadratic Regulator Design for Nonlinear Biological Movement Systems*. ICINCO.
   - Tassa, Y., Erez, T., & Todorov, E. (2012). *Synthesis and Stabilization of Complex Behaviors through Online Trajectory Optimization*. IROS.

2. **增广拉格朗日法**:
   - Bertsekas, D. P. (1982). *Constrained Optimization and Lagrange Multiplier Methods*. Academic Press.
   - Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization* (2nd ed.). Springer.

3. **LQR 理论**:
   - Åström, K. J., & Murray, R. M. (2008). *Feedback Systems: An Introduction for Scientists and Engineers*. Princeton University Press.
   - Lewis, F. L., Vrabie, D., & Syrmos, V. L. (2012). *Optimal Control* (3rd ed.). Wiley.

### 在线资源

- **UC Berkeley CS 287**: Advanced Robotics (Pieter Abbeel)
  - Lecture on LQR and iLQR: https://people.eecs.berkeley.edu/~pabbeel/cs287-fa19/

- **CMU 16-745**: Optimal Control (Zac Manchester)
  - iLQR Tutorial: https://optimalcontrolclass.github.io/

- **MuJoCo Physics Engine**:
  - 包含高效的 iLQR 实现: https://mujoco.org/

### 代码参考

- **Drake (MIT)**:
  - C++ 机器人工具箱，包含 iLQR 实现
  - https://drake.mit.edu/

- **Crocoddyl (LAAS-CNRS)**:
  - 专门的微分动态规划 (DDP) 库，与 iLQR 相关
  - https://github.com/loco-3d/crocoddyl

---

## 总结

`linearizedInitialGuess()` 方法是 iLQR 优化的关键起点:

**核心思想**:
- 利用 LQR 理论在目标轨迹附近快速生成高质量初始猜测
- 通过离散时间 Riccati 方程计算反馈增益
- 使用非线性动力学仿真生成实际轨迹

**数学基础**:
- 动态规划和 Bellman 最优性原理
- 值函数的二次近似
- 线性二次调节器理论

**实现技巧**:
- 在目标状态和零控制处线性化 (简化)
- 使用固定正则化避免数值问题
- 混合策略: 线性控制器 + 非线性仿真

**应用价值**:
- 加速 iLQR 收敛 (减少迭代次数)
- 提高优化鲁棒性 (避免局部最优)
- 适用于轨迹跟踪、稳定控制、MPC 等场景

通过深入理解这一方法，可以更好地调试和改进 iLQR 求解器，适应不同的应用需求。
