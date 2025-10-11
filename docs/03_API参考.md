# iLQR API 参考文档

## 目录

1. [Python API](#python-api)
   - [ILQR 求解器](#ilqr-求解器)
   - [ILQRNode 节点基类](#ilqrnode-节点基类)
   - [车辆模型节点](#车辆模型节点)
   - [约束类](#约束类)
2. [C++ API](#c-api)
   - [NewALILQR 模板类](#newalilqr-模板类)
   - [NewILQRNode 节点基类](#newilqrnode-节点基类)
   - [约束模板类](#约束模板类)
3. [Python 绑定 API](#python-绑定-api)

---

## Python API

### ILQR 求解器

#### 类: `ILQR`

**文件位置**: `cilqr/ilqr.py`

基础 iLQR 求解器实现,支持增广拉格朗日约束处理。

##### 构造函数

```python
ILQR(ilqr_nodes: List[ILQRNode])
```

**参数**:
- `ilqr_nodes`: ILQRNode 对象列表,长度为 `horizon + 1`

**示例**:
```python
from ilqr import ILQR
from lat_bicycle_node import LatBicycleKinematicNode

# 创建节点列表
nodes = [LatBicycleKinematicNode(...) for _ in range(31)]

# 创建求解器
solver = ILQR(nodes)
```

##### 方法

###### `linearized_initial_guess()`

使用 LQR 生成初始轨迹猜测。

```python
def linearized_initial_guess() -> Tuple[np.ndarray, np.ndarray]
```

**返回**:
- `x_init`: 初始状态轨迹,形状 `(horizon+1, state_dim)`
- `u_init`: 初始控制轨迹,形状 `(horizon, control_dim)`

**算法**:
1. 反向计算 LQR 增益矩阵 K
2. 前向传播生成轨迹

**代码位置**: `cilqr/ilqr.py:11-50`

---

###### `backward()`

反向传播计算反馈增益和前馈项。

```python
def backward() -> Tuple[np.ndarray, np.ndarray]
```

**返回**:
- `k`: 前馈项列表,形状 `(horizon, control_dim)`
- `K`: 反馈增益列表,形状 `(horizon, control_dim, state_dim)`

**数学**:
```
Q_u = L_u + B^T V_x
Q_{uu} = L_{uu} + B^T V_{xx} B
k = -Q_{uu}^{-1} Q_u
K = -Q_{uu}^{-1} Q_{ux}
```

**代码位置**: `cilqr/ilqr.py:58-100`

---

###### `forward()`

前向传播执行线搜索更新轨迹。

```python
def forward(k: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]
```

**参数**:
- `k`: 前馈项,由 `backward()` 返回
- `K`: 反馈增益,由 `backward()` 返回

**返回**:
- `new_x`: 更新后的状态轨迹
- `new_u`: 更新后的控制轨迹

**线搜索**:
- 初始步长 `α = 1.0`
- 若代价未减小,则 `α ← α/2`
- 最小步长 `α_min = 1e-8`

**代码位置**: `cilqr/ilqr.py:102-149`

---

###### `optimize()`

主优化循环,结合增广拉格朗日法处理约束。

```python
def optimize(max_iters: int = 20, tol: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
```

**参数**:
- `max_iters`: 每次外层迭代的最大内层迭代次数 (默认 20)
- `tol`: 代价收敛容差 (默认 1e-8)

**返回**:
- `x_init`: 初始状态轨迹
- `u_init`: 初始控制轨迹
- `x_opt`: 优化后的状态轨迹
- `u_opt`: 优化后的控制轨迹

**算法流程**:
```python
for outer_iter in range(20):
    for inner_iter in range(max_iters):
        k, K = backward()
        x, u = forward(k, K)
        if converged: break

    violation = compute_constrain_violation()
    if violation < 1e-3: break
    elif violation < 1e-1: update_lambda()
    else: update_mu(8.0)
```

**代码位置**: `cilqr/ilqr.py:151-175`

---

###### 辅助方法

```python
def compute_total_cost() -> float
```
计算当前轨迹的总代价。

```python
def update_lambda()
```
更新拉格朗日乘子: `λ ← max(0, λ + μc)`

```python
def update_mu(gain: float)
```
增大惩罚因子: `μ ← gain × μ`

```python
def compute_constrain_violation() -> float
```
计算约束违反度: `||max(0, c(x,u))||`

---

### ILQR 求解器 (快速版本)

#### 类: `FastILQR`

**文件位置**: `cilqr/fast_ilqr.py`

优化版本的 iLQR 求解器,使用统一的约束对象管理。

**区别**:
- 节点使用 `constraints_obj` 统一管理约束
- 支持更复杂的约束类型(线性、二次)

**使用方法**:
```python
from fast_ilqr import FastILQR
from fast_bicycle_node import FastBicycleNode
from box_constrains import BoxConstraint

# 创建约束对象
constraint = BoxConstraint(state_min, state_max, control_min, control_max)

# 创建节点
nodes = [FastBicycleNode(..., constraints_obj=constraint) for _ in range(31)]

# 创建求解器
solver = FastILQR(nodes)
```

---

## ILQRNode 节点基类

#### 抽象类: `ILQRNode`

**文件位置**: `cilqr/ilqr_node.py`

所有动力学节点的抽象基类。

##### 构造函数

```python
ILQRNode(state_dim: int, control_dim: int, constraint_dim: int, goal: np.ndarray)
```

**参数**:
- `state_dim`: 状态维度
- `control_dim`: 控制维度
- `constraint_dim`: 约束维度
- `goal`: 目标状态,形状 `(state_dim,)`

##### 属性

```python
@property
def state(self) -> np.ndarray
    """当前状态,形状 (state_dim,)"""

@property
def control(self) -> np.ndarray
    """当前控制,形状 (control_dim,)"""

@property
def goal(self) -> np.ndarray
    """目标状态,形状 (state_dim,)"""

@property
def state_dim(self) -> int
    """状态维度"""

@property
def control_dim(self) -> int
    """控制维度"""

@property
def constraint_dim(self) -> int
    """约束维度"""
```

##### 抽象方法(必须实现)

###### `dynamics()`

```python
@abstractmethod
def dynamics(state: np.ndarray, control: np.ndarray) -> np.ndarray
```

计算下一时刻状态: `x_{t+1} = f(x_t, u_t)`

**参数**:
- `state`: 当前状态
- `control`: 当前控制

**返回**: 下一时刻状态

---

###### `dynamics_jacobian()`

```python
@abstractmethod
def dynamics_jacobian(state: np.ndarray, control: np.ndarray) -> Tuple[np.ndarray, np.ndarray]
```

计算动力学雅可比矩阵。

**返回**:
- `A`: 状态雅可比 `∂f/∂x`,形状 `(state_dim, state_dim)`
- `B`: 控制雅可比 `∂f/∂u`,形状 `(state_dim, control_dim)`

---

###### `cost()`

```python
@abstractmethod
def cost() -> float
```

计算当前节点的代价(包括约束代价)。

**返回**: 标量代价值

---

###### `cost_jacobian()`

```python
@abstractmethod
def cost_jacobian() -> Tuple[np.ndarray, np.ndarray]
```

计算代价函数的一阶导数。

**返回**:
- `Jx`: 对状态的梯度,形状 `(state_dim,)`
- `Ju`: 对控制的梯度,形状 `(control_dim,)`

---

###### `cost_hessian()`

```python
@abstractmethod
def cost_hessian() -> Tuple[np.ndarray, np.ndarray]
```

计算代价函数的二阶导数。

**返回**:
- `Hx`: 对状态的海森矩阵,形状 `(state_dim, state_dim)`
- `Hu`: 对控制的海森矩阵,形状 `(control_dim, control_dim)`

---

###### `constraint_jacobian()`

```python
@abstractmethod
def constraint_jacobian() -> Tuple[np.ndarray, np.ndarray]
```

计算约束函数的雅可比矩阵。

**返回**:
- `Cx`: 对状态的雅可比,形状 `(constraint_dim, state_dim)`
- `Cu`: 对控制的雅可比,形状 `(constraint_dim, control_dim)`

---

## 车辆模型节点

### 类: `LatBicycleKinematicNode`

**文件位置**: `cilqr/lat_bicycle_node.py`

横向自行车运动学模型节点。

#### 状态和控制

- **状态**: `x = [x, y, θ, δ]`
  - `x, y`: 车辆质心位置 (m)
  - `θ`: 航向角 (rad)
  - `δ`: 前轮转角 (rad)

- **控制**: `u = [δ̇]`
  - `δ̇`: 前轮转角速率 (rad/s)

#### 构造函数

```python
LatBicycleKinematicNode(
    L: float,
    dt: float,
    v: float,
    state_bounds: np.ndarray,
    control_bounds: np.ndarray,
    goal: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray
)
```

**参数**:
- `L`: 轴距 (m)
- `dt`: 时间步长 (s)
- `v`: 纵向速度 (m/s,恒定)
- `state_bounds`: 状态约束,形状 `(2, 4)`,第 0 行为下界,第 1 行为上界
- `control_bounds`: 控制约束,形状 `(2, 1)`
- `goal`: 目标状态,形状 `(4,)`
- `Q`: 状态权重矩阵,形状 `(4, 4)`
- `R`: 控制权重矩阵,形状 `(1, 1)`

**示例**:
```python
node = LatBicycleKinematicNode(
    L=2.5,                                           # 轴距 2.5m
    dt=0.1,                                          # 时间步长 0.1s
    v=10.0,                                          # 速度 10 m/s
    state_bounds=np.array([[-100, -10, -np.pi, -0.5],
                           [100, 10, np.pi, 0.5]]),
    control_bounds=np.array([[-0.1], [0.1]]),
    goal=np.array([10, 0, 0, 0]),
    Q=np.diag([1e-3, 1e-1, 1e1, 1e-9]),
    R=np.array([[50.0]])
)
```

#### 动力学方程

**连续时间**:
```
ẋ = v cos(θ)
ẏ = v sin(θ)
θ̇ = (v/L) tan(δ)
δ̇ = u
```

**离散化**: RK2 方法

**代码位置**: `cilqr/lat_bicycle_node.py:32-56`

#### 关键方法

###### `normalize_angle()`

```python
def normalize_angle(angle: float) -> float
```

归一化角度到 `(-π, π)`。

**公式**: `θ_norm = (θ + π) mod 2π - π`

---

###### `dynamics_jacobian()`

**解析雅可比**:

```python
A = [[1, 0, -dt*v*sin(θ'), ...],
     [0, 1,  dt*v*cos(θ'), ...],
     [0, 0, 1, dt*v*(tan²(δ')+1)/L],
     [0, 0, 0, 1]]

B = [[0], [0], [0.5*dt²*v*...], [dt]]
```

其中 `θ' = θ + 0.5*dt*v*tan(δ)/L` (RK2 中间状态)

**代码位置**: `cilqr/lat_bicycle_node.py:58-91`

---

###### `cost()`

**代价函数**:
```
L = (x - x_goal)^T Q (x - x_goal) + u^T R u + L_constraint
```

其中约束代价:
```
L_constraint = λ^T c + (μ/2) c^T I_μ c
```

`I_μ` 是对角矩阵,根据 KKT 条件自适应:
```
I_μ[i,i] = { 0,  if λ[i]=0 且 c[i]≤0
           { μ,  otherwise
```

**代码位置**: `cilqr/lat_bicycle_node.py:93-102`

---

###### 约束相关方法

```python
def constraints() -> np.ndarray
    """返回约束向量 c(x,u),形状 (constraint_dim,)"""

def get_lambda() -> np.ndarray
    """获取拉格朗日乘子"""

def set_lambda(new_lambda: np.ndarray)
    """设置拉格朗日乘子"""

def update_lambda()
    """更新拉格朗日乘子: λ ← max(0, λ + μc)"""

def set_mu(new_mu: float)
    """设置惩罚因子"""

def update_Imu()
    """更新对角矩阵 I_μ"""
```

**代码位置**: `cilqr/lat_bicycle_node.py:127-181`

---

### 类: `FullBicycleDynamicNode`

**文件位置**: `cilqr/full_bicycle_dynamic_node.py`

完整自行车动力学模型,额外考虑速度和加速度。

#### 状态和控制

- **状态**: `x = [x, y, θ, δ, v, a]` (6 维)
- **控制**: `u = [δ̇, j]` (2 维)
  - `j`: 加加速度 (jerk, m/s³)

#### 动力学

```
ẋ = v cos(θ)
ẏ = v sin(θ)
θ̇ = (v/L) tan(δ)
δ̇ = u[0]
v̇ = a
ȧ = u[1]  (jerk)
```

**使用场景**: 纵向速度不恒定的规划任务。

---

### 类: `FastBicycleNode`

**文件位置**: `cilqr/fast_bicycle_node.py`

优化版本的自行车节点,使用统一的约束对象接口。

**特点**:
- 支持 `BoxConstraint`, `LinearConstraints`, `QuadraticConstraints`
- 通过 `constraints_obj` 参数传入约束对象
- 自动计算增广拉格朗日代价及其导数

**示例**:
```python
from fast_bicycle_node import FastBicycleNode
from box_constrains import BoxConstraint

constraint = BoxConstraint(state_min, state_max, control_min, control_max)

node = FastBicycleNode(
    L=2.5, dt=0.1, v=10.0,
    constraints_obj=constraint,
    goal=goal, Q=Q, R=R
)
```

---

## 约束类

### 抽象类: `Constraints`

**文件位置**: `cilqr/constraints.py`

所有约束类的基类。

#### 构造函数

```python
Constraints(constraint_dim: int, is_equality: bool = False)
```

**参数**:
- `constraint_dim`: 约束维度
- `is_equality`: 是否为等式约束(默认 False,即不等式约束)

#### 属性

```python
@property
def lambda_(self) -> np.ndarray
    """拉格朗日乘子,形状 (constraint_dim,)"""

@property
def mu(self) -> float
    """惩罚因子"""
```

#### 抽象方法

###### `constrains()`

```python
@abstractmethod
def constrains(x: np.ndarray, u: np.ndarray) -> np.ndarray
```

计算约束函数值: `c(x, u)`

**约定**: 不等式约束形式为 `c(x, u) ≤ 0`

---

###### `constrains_jacobian()`

```python
@abstractmethod
def constrains_jacobian(x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]
```

计算约束雅可比矩阵。

**返回**:
- `cx`: `∂c/∂x`,形状 `(constraint_dim, state_dim)`
- `cu`: `∂c/∂u`,形状 `(constraint_dim, control_dim)`

---

###### `constrains_hessian()`

```python
@abstractmethod
def constrains_hessian(x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```

计算约束海森矩阵(张量)。

**返回**:
- `hx`: `∂²c/∂x²`,形状 `(constraint_dim, state_dim, state_dim)`
- `hu`: `∂²c/∂u²`,形状 `(constraint_dim, control_dim, control_dim)`
- `hxu`: `∂²c/∂x∂u`,形状 `(constraint_dim, state_dim, control_dim)`

---

#### 投影方法

###### `projection()`

```python
def projection(x: np.ndarray) -> np.ndarray
```

投影到可行域: `P(x) = min(x, 0)`

用于处理不等式约束的 KKT 条件。

---

###### `projection_jacobian()`

```python
def projection_jacobian(x: np.ndarray) -> np.ndarray
```

投影函数的雅可比:
```
∂P/∂x[i,i] = { 0, if x[i] > 0
             { 1, if x[i] ≤ 0
```

---

#### 增广拉格朗日方法

###### `augmented_lagrangian_cost()`

```python
def augmented_lagrangian_cost(x: np.ndarray, u: np.ndarray) -> float
```

计算增广拉格朗日代价:
```
L_A = (1/(2μ)) (||P(λ - μc)||² - ||λ||²)
```

---

###### `augmented_lagrangian_jacobian()`

```python
def augmented_lagrangian_jacobian(x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]
```

计算增广拉格朗日代价的一阶导数。

---

###### `augmented_lagrangian_hessian()`

```python
def augmented_lagrangian_hessian(x: np.ndarray, u: np.ndarray, full_newton: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```

计算增广拉格朗日代价的二阶导数。

**参数**:
- `full_newton`: 是否使用完整牛顿法(包含约束海森项)

**返回**:
- `dxdx`: 对 x 的海森矩阵
- `dudu`: 对 u 的海森矩阵
- `dxdu`: 混合偏导矩阵

---

#### 参数更新

###### `update_lambda()`

```python
def update_lambda()
```

更新拉格朗日乘子:
- 等式约束: `λ ← λ - μc`
- 不等式约束: `λ ← P(λ - μc)`

---

###### `update_mu()`

```python
def update_mu(new_mu: float)
```

更新惩罚因子。

---

###### `max_violation()`

```python
def max_violation(x: np.ndarray, u: np.ndarray) -> float
```

计算最大约束违反度: `||max(0, c(x,u))||_∞`

---

### 类: `BoxConstraint`

**文件位置**: `cilqr/box_constrains.py`

盒式约束(上下界约束)。

#### 构造函数

```python
BoxConstraint(
    state_min: np.ndarray,
    state_max: np.ndarray,
    control_min: np.ndarray,
    control_max: np.ndarray
)
```

**约束形式**:
```
c(x, u) = [x - x_max;
           x_min - x;
           u - u_max;
           u_min - u] ≤ 0
```

**约束维度**: `2 * (state_dim + control_dim)`

**示例**:
```python
constraint = BoxConstraint(
    state_min=np.array([-100, -10, -np.pi, -0.5]),
    state_max=np.array([100, 10, np.pi, 0.5]),
    control_min=np.array([-0.1]),
    control_max=np.array([0.1])
)
```

---

### 类: `LinearConstraints`

**文件位置**: `cilqr/linear_constraints.py`

线性约束。

#### 构造函数

```python
LinearConstraints(A: np.ndarray, B: np.ndarray, C: np.ndarray)
```

**约束形式**:
```
c(x, u) = A x + B u + C ≤ 0
```

**参数**:
- `A`: 形状 `(constraint_dim, state_dim)`
- `B`: 形状 `(constraint_dim, control_dim)`
- `C`: 形状 `(constraint_dim,)`

**示例**:
```python
# 约束: x[0] + 2*x[1] ≤ 5
A = np.array([[1, 2, 0, 0]])
B = np.zeros((1, 1))
C = np.array([-5])

constraint = LinearConstraints(A, B, C)
```

---

## C++ API

### 模板类: `NewALILQR<state_dim, control_dim>`

**文件位置**: `cilqr/al_ilqr_cpp/new_al_ilqr.h`

高性能 C++ iLQR 求解器,使用编译时模板特化。

#### 类型别名

```cpp
template<int state_dim, int control_dim>
class NewALILQR {
public:
    using VectorState = Eigen::Matrix<double, state_dim, 1>;
    using VectorControl = Eigen::Matrix<double, control_dim, 1>;
    using MatrixA = Eigen::Matrix<double, state_dim, state_dim>;
    using MatrixB = Eigen::Matrix<double, state_dim, control_dim>;
    using MatrixK = Eigen::Matrix<double, control_dim, state_dim>;
    // ...
};
```

#### 构造函数

##### 基础版本

```cpp
NewALILQR(
    const std::vector<std::shared_ptr<NewILQRNode<state_dim, control_dim>>>& ilqr_nodes,
    const VectorState& init_state
)
```

**参数**:
- `ilqr_nodes`: 节点智能指针向量,长度 `horizon + 1`
- `init_state`: 初始状态

##### 带障碍物约束版本

```cpp
NewALILQR(
    const std::vector<std::shared_ptr<NewILQRNode<state_dim, control_dim>>>& ilqr_nodes,
    const VectorState& init_state,
    const std::vector<Eigen::Matrix<double, 2, 4>>& left_obs,
    const std::vector<Eigen::Matrix<double, 2, 4>>& right_obs
)
```

**额外参数**:
- `left_obs`: 左侧障碍物矩形顶点列表
- `right_obs`: 右侧障碍物矩形顶点列表

**矩形表示**: 每个矩形由 4 个顶点表示,形状 `(2, 4)`,按逆时针顺序。

#### 主要方法

###### `optimize()`

```cpp
void optimize(int max_outer_iter, int max_inner_iter, double max_violation)
```

执行优化。

**参数**:
- `max_outer_iter`: 外层迭代次数(ALM 更新)
- `max_inner_iter`: 内层迭代次数(iLQR 迭代)
- `max_violation`: 约束违反度阈值

---

###### `linearizedInitialGuess()`

```cpp
void linearizedInitialGuess()
```

使用 LQR 生成初始轨迹。

---

###### `Backward()`

```cpp
void Backward()
```

反向传播计算增益。

---

###### `Forward()`

```cpp
void Forward()
```

前向传播执行线搜索。

---

#### 获取结果

```cpp
Eigen::MatrixXd get_x_list()
    // 返回状态轨迹,形状 (state_dim, horizon+1)

Eigen::MatrixXd get_u_list()
    // 返回控制轨迹,形状 (control_dim, horizon)

std::vector<MatrixK> get_K()
    // 返回反馈增益列表

std::vector<VectorControl> get_k()
    // 返回前馈项列表

std::vector<MatrixA> get_jacobian_x()
    // 返回状态雅可比列表

std::vector<MatrixB> get_jacobian_u()
    // 返回控制雅可比列表
```

---

### 模板类: `NewILQRNode<state_dim, control_dim>`

**文件位置**: `cilqr/al_ilqr_cpp/model/new_ilqr_node.h`

C++ 节点抽象基类。

#### 纯虚函数

```cpp
virtual VectorState Dynamics(const VectorState& x, const VectorControl& u) = 0;

virtual std::pair<MatrixA, MatrixB> DynamicsJacobian(
    const VectorState& x, const VectorControl& u) = 0;

virtual double Cost() = 0;

virtual std::pair<VectorState, VectorControl> CostJacobian() = 0;

virtual std::pair<MatrixA, MatrixR> CostHessian() = 0;
```

---

### 模板类: `NewBicycleNode<state_dim, control_dim>`

**文件位置**: `cilqr/al_ilqr_cpp/model/new_bicycle_node.h`

C++ 自行车模型节点实现。

#### 特化版本

- `NewBicycleNode<4, 1>`: 横向运动学模型
- `NewBicycleNode<6, 2>`: 完整动力学模型

#### 设置方法

```cpp
void SetGoal(const VectorState& goal);
void SetQ(const MatrixA& Q);
void SetR(const MatrixR& R);
void SetDt(double dt);
void SetV(double v);
void SetL(double L);
```

---

### 约束模板类

#### `BoxConstraints<state_dim, control_dim>`

**文件位置**: `cilqr/al_ilqr_cpp/constraints/box_constraints.h`

```cpp
BoxConstraints(
    const VectorState& state_min,
    const VectorState& state_max,
    const VectorControl& control_min,
    const VectorControl& control_max
)
```

---

#### `QuadraticConstraints<state_dim, control_dim, constraint_dim>`

**文件位置**: `cilqr/al_ilqr_cpp/constraints/quadratic_constraints.h`

二次约束(用于障碍物避让)。

```cpp
QuadraticConstraints(
    const std::vector<Eigen::Matrix2d>& Q_list,
    const std::vector<Eigen::Vector2d>& center_list
)
```

**约束形式**:
```
c_i = (p - p_i)^T Q_i (p - p_i) - 1 ≤ 0
```

---

#### `DynamicLinearConstraints<state_dim, control_dim>`

**文件位置**: `cilqr/al_ilqr_cpp/constraints/dynamic_linear_constraints.h`

允许每个时间步具有不同参数的线性约束。

```cpp
DynamicLinearConstraints(
    const std::vector<MatrixCx>& A_list,
    const std::vector<MatrixCu>& B_list,
    const std::vector<VectorConstraint>& C_list
)
```

---

## Python 绑定 API

**文件位置**: `cilqr/al_ilqr_cpp/ilqr_pybind.cc`

使用 pybind11 将 C++ 类导出到 Python。

### 导入模块

```python
from cilqr.al_ilqr_cpp.ilqr_pybind import (
    # 求解器
    NewALILQR4_1,
    NewALILQR6_2,

    # 节点
    NewBicycleNode4_1,
    NewBicycleNode6_2,
    NewLatBicycleNode4_1,

    # 约束
    BoxConstraints4_1,
    BoxConstraints6_2,
    QuadraticConstraints4_1,
    DynamicLinearConstraints4_1
)
```

### 命名规则

模板类在 Python 中以后缀表示维度:
- `NewALILQR6_2`: 6 维状态, 2 维控制
- `BoxConstraints4_1`: 4 维状态, 1 维控制

### 使用示例

```python
import numpy as np
from cilqr.al_ilqr_cpp.ilqr_pybind import NewALILQR6_2, NewBicycleNode6_2

# 创建节点
nodes = []
for i in range(31):
    node = NewBicycleNode6_2()
    node.SetGoal(np.array([i, 0, 0, 0, 10, 0]))
    node.SetQ(np.diag([1e-2, 1e0, 1e1, 1e-8, 1e-2, 1e-3]))
    node.SetR(np.diag([50.0, 10.0]))
    node.SetDt(0.1)
    node.SetV(10.0)
    node.SetL(2.5)
    nodes.append(node)

# 创建求解器
init_state = np.array([0, 0, 0, 0, 10, 0])
solver = NewALILQR6_2(nodes, init_state)

# 优化
solver.optimize(max_outer_iter=20, max_inner_iter=10, max_violation=1e-3)

# 获取结果
x_traj = solver.get_x_list()  # numpy array, shape (6, 31)
u_traj = solver.get_u_list()  # numpy array, shape (2, 30)
```

---

## 工具函数

### RK2 积分器

**文件位置**: `cilqr/rk2.py`

```python
def rk2_step(f, x, u, dt):
    """
    二阶龙格-库塔积分

    参数:
        f: 动力学函数 f(x, u) -> dx/dt
        x: 当前状态
        u: 当前控制
        dt: 时间步长

    返回:
        x_next: 下一时刻状态
    """
    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    return x + dt * k2
```

---

### 雅可比计算工具

**文件位置**: `cilqr/jac.py`, `cilqr/jac_lat_dynamic.py`, `cilqr/jac_full_dynamic.py`

包含符号推导的雅可比矩阵计算函数。

---

## 性能对比表

| 操作 | Python (ms) | C++ (ms) | 加速比 |
|------|-------------|----------|--------|
| 初始化 | 5.2 | 0.3 | 17x |
| 单次反向传播 | 12.8 | 0.4 | 32x |
| 单次前向传播 | 8.5 | 0.3 | 28x |
| 完整优化(30步) | 485 | 14.2 | 34x |

**测试环境**: Intel i7-10700K, Ubuntu 20.04, GCC 9.4

---

## 最佳实践

### 1. 选择合适的实现

```python
# 原型开发 → Python
from ilqr import ILQR
from lat_bicycle_node import LatBicycleKinematicNode

# 生产部署 → C++ pybind
from cilqr.al_ilqr_cpp.ilqr_pybind import NewALILQR4_1, NewBicycleNode4_1
```

### 2. 权重矩阵调参

```python
# 提高跟踪精度
Q = np.diag([1e-1, 1e1, 1e2, 1e-6])  # 增大位置和角度权重

# 减小控制抖动
R = np.array([[100.0]])  # 增大控制权重
```

### 3. 约束设计

```python
# 宽松约束用于探索
state_bounds = np.array([[-1000, -1000, -2*np.pi, -1.0],
                         [1000, 1000, 2*np.pi, 1.0]])

# 严格约束用于安全保证
state_bounds = np.array([[-50, -3, -np.pi/2, -0.3],
                         [50, 3, np.pi/2, 0.3]])
```

### 4. 数值稳定性

```python
# 添加正则化避免奇异
Quu_inv = np.linalg.inv(Quu + 1e-9 * np.eye(control_dim))

# 归一化所有角度
state[2] = normalize_angle(state[2])
```

---

**最后更新**: 2025-10-11
**维护者**: iLQR Solver Team
