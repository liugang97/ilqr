# CLAUDE.md

这个文件为 Claude Code (claude.ai/code) 在这个代码库中工作时提供指导。

## 项目概述

这是一个 **iLQR (iterative Linear Quadratic Regulator) 求解器**项目,使用 C++ 和 Eigen 实现,通过 pybind11 提供 Python 接口。项目采用增广拉格朗日方法 (Augmented Lagrangian) 处理约束优化问题,主要用于车辆轨迹规划和控制。

## 核心架构

### 1. 三层架构设计

- **约束层** (`cilqr/al_ilqr_cpp/constraints/`): 定义各种约束类型
  - `Constraints`: 基础约束接口
  - `LinearConstraints`: 线性约束 (B*u + A*x <= C)
  - `BoxConstraints`: 盒式约束 (状态和控制的上下界)
  - `QuadraticConstraints`: 二次约束 (用于障碍物避障,如圆形障碍物)
  - `DynamicLinearConstraints`: 动态线性约束

- **模型层** (`cilqr/al_ilqr_cpp/model/`): 定义车辆动力学模型
  - `NewILQRNode`: iLQR 节点基类,封装代价函数、动力学和约束
  - `NewBicycleNode`: 完整动力学自行车模型 (6维状态: [x, y, θ, δ, v, a], 2维控制: [δ_rate, a_rate])
  - `NewLatBicycleNode`: 横向自行车模型 (4维状态)
  - `parallel_compution_function.h`: 并行计算优化

- **求解器层** (`cilqr/al_ilqr_cpp/`): 核心 iLQR 算法
  - `new_al_ilqr.h`: 增广拉格朗日 iLQR 求解器,实现双层优化:
    - 外层循环: 更新拉格朗日乘子 λ 和惩罚系数 μ
    - 内层循环: 标准 iLQR (Backward-Forward 迭代)
  - 支持并行线搜索优化 (`ParallelLinearSearch`)

### 2. Python 绑定架构

`ilqr_pybind.cc` 使用 pybind11 将 C++ 类暴露给 Python:
- 约束绑定: `bind_constraints`, `bind_box_constraints`, `bind_quadratic_constraints`
- 节点绑定: `bind_new_bicycle_node`, `bind_new_lat_bicycle_node`
- 求解器绑定: `bind_new_al_ilqr`

命名约定: `ClassName<state_dim>_<control_dim>_<num_constraints>`
- 例如: `BoxConstraints6_2` (6维状态, 2维控制)
- 例如: `QuadraticConstraints6_2_5` (6维状态, 2维控制, 5个约束)

## 构建系统

项目使用 **Bazel** 构建系统:

### 主要构建命令

```bash
# 进入 C++ 代码目录
cd cilqr/al_ilqr_cpp

# 构建 Python 绑定模块 (生成 ilqr_pybind.so)
bazel build //:ilqr_pybind --config=opt

# 构建 C++ 测试程序
bazel build //:test_new_al_ilqr
bazel build //:test_lat_al_ilqr
bazel build //:test_dynamic_constraints_ilqr

# 运行 C++ 测试
bazel run //:test_new_al_ilqr
```

### 构建配置

- 优化标志: `-O3 -march=native -DEIGEN_VECTORIZE`
- 编译调试版本: `bazel build //:ilqr_pybind -c dbg --copt=-g`
- 清除缓存: `bazel clean --expunge`

### 依赖项

- **Eigen 3.3.7**: 线性代数库
- **pybind11 2.13.6**: Python-C++ 绑定
- 依赖在 `WORKSPACE` 文件中通过 `http_archive` 管理

### 生成的输出

- Python 模块: `bazel-bin/ilqr_pybind.so`
- Python 代码需要将此路径添加到 `sys.path`:
  ```python
  sys.path.append("/path/to/cilqr/al_ilqr_cpp/bazel-bin")
  import ilqr_pybind
  ```

## Python 测试

### 主要测试文件

- `cilqr/test_pybind.py`: 完整的 Python 绑定测试,演示:
  1. 生成 S 形参考轨迹
  2. 盒式约束优化
  3. 带障碍物的二次约束优化
  4. 结果可视化对比

### 运行 Python 测试

```bash
# 确保已构建 ilqr_pybind.so
cd cilqr/al_ilqr_cpp
bazel build //:ilqr_pybind

# 返回项目根目录运行测试
cd ../..
python cilqr/test_pybind.py
```

## 关键实现细节

### iLQR 算法流程

1. **初始化** (`linearizedInitialGuess`): 使用 LQR 生成初始轨迹
2. **外层迭代**: 增广拉格朗日法更新约束处理
   - 计算约束违反 (`ComputeConstraintViolation`)
   - 更新拉格朗日乘子 λ (`UpdateLambda`)
   - 更新惩罚系数 μ (`UpdateMu`)
3. **内层迭代** (`ILQRProcess`):
   - 更新动态约束 (`UpdateConstraints`)
   - 计算导数 (`CalcDerivatives`): 代价函数和动力学的雅可比/海森矩阵
   - 向后传播 (`Backward`): 计算反馈增益 K 和前馈项 k
   - 向前传播 (`Forward`): 线搜索更新轨迹

### 数值优化技巧

- **RK2 积分**: 使用二阶 Runge-Kutta 方法离散化连续动力学
- **角度归一化**: 自动处理 θ 和 δ 的周期性
- **并行线搜索**: 同时评估多个步长 (通过 `PARALLEL_NUM` 定义)
- **正则化**: 动力学模型中添加小量 k 防止数值问题

### 约束处理

- 盒式约束: 直接通过上下界限制
- 二次约束: 用于圆形障碍物,形式为 `x^T*Q*x + A^T*x + C <= 0`
- 动态约束: 根据当前轨迹动态添加约束 (如避障)

## 开发注意事项

### 添加新的车辆模型

1. 在 `model/` 下创建新的 node 类,继承 `NewILQRNode<state_dim, control_dim>`
2. 实现 `dynamics()` 和 `dynamics_jacobian()` 方法
3. 在 `model/BUILD` 中添加 cc_library
4. 在 `ilqr_pybind.cc` 中添加绑定

### 添加新的约束类型

1. 在 `constraints/` 下创建新类,继承 `Constraints<state_dim, control_dim, num_constraints>`
2. 实现约束评估和梯度计算
3. 在 `constraints/BUILD` 中添加 cc_library
4. 在 `constraints_bind.h` 中添加绑定函数

### 调试技巧

- 编译调试版本: `bazel build //:ilqr_pybind -c dbg --copt=-g`
- 取消注释 `new_al_ilqr.h` 中的时间测量代码以分析性能
- 使用 `std::cout` 输出中间变量 (代码中已有注释的示例)

## 典型使用流程

```python
# 1. 创建约束对象
constraints = ilqr_pybind.BoxConstraints6_2(state_min, state_max, control_min, control_max)

# 2. 创建节点列表 (每个时间步一个节点)
nodes = []
for i in range(horizon + 1):
    node = ilqr_pybind.NewBicycleNodeBoxConstraints6_2(L, dt, k, goal_states[i], Q, R, constraints)
    nodes.append(node)

# 3. 创建求解器
solver = ilqr_pybind.NewALILQR6_2(nodes, init_state)

# 4. 执行优化
solver.optimize(max_outer_iter, max_inner_iter, max_violation)

# 5. 获取结果
x_list = solver.get_x_list()  # 状态轨迹 (state_dim × horizon+1)
u_list = solver.get_u_list()  # 控制序列 (control_dim × horizon)
```
