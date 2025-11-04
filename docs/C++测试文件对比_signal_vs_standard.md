# test_new_al_ilqr_signal.cc vs test_new_al_ilqr.cc 对比分析

## 概述

这两个 C++ 测试文件虽然都使用了相同的基础设置（S 形轨迹、车辆模型），但它们的**测试目的完全不同**：

- **`test_new_al_ilqr_signal.cc`**: **性能基准测试**（Benchmark）
- **`test_new_al_ilqr.cc`**: **功能完整性测试**（Functional Test）

---

## 详细对比表

| 对比项 | test_new_al_ilqr_signal.cc | test_new_al_ilqr.cc |
|--------|----------------------------|---------------------|
| **主要目的** | 性能测试、算法对比 | 功能测试、约束验证 |
| **优化次数** | 0 次（不执行完整优化） | 2 次（盒式约束 + 二次约束） |
| **测试内容** | 单步 Backward/Forward | 完整 AL-iLQR 优化流程 |
| **障碍物** | 无 | 矩形障碍物 + 圆形障碍物 |
| **约束类型** | 仅盒式约束 | 盒式 + 二次约束 + 动态线性约束 |
| **时间测量** | ✅ 详细性能计时 | ❌ 无性能测量 |
| **输出内容** | 性能数据、数值对比 | 优化结果轨迹 |
| **代码行数** | 122 行 | 146 行 |

---

## 核心功能差异

### test_new_al_ilqr_signal.cc - 性能测试

#### 测试流程（第 71-117 行）

```cpp
solver.linearizedInitialGuess();    // 1. 生成初始猜测
solver.CalcDerivatives();           // 2. 计算导数
solver.Backward();                  // 3. 向后传播

// 计时开始
auto forward_start = std::chrono::high_resolution_clock::now();
solver.Forward();                   // 4. 向前传播（计时）
auto forward_end = std::chrono::high_resolution_clock::now();

// ... 更多性能测试
```

**不执行完整优化**，只测试单次迭代的性能。

#### 性能测试项目

**1. 并行雅可比计算性能**（第 85-117 行）

```cpp
// 多状态并行计算雅可比矩阵
auto multi_start_jacobian = std::chrono::high_resolution_clock::now();
auto jacobian = ParallelFullBicycleDynamicJacobianRK2<50>(
    x_list.block(0,0,6,50),
    u_list.block(0,0,2,50),
    0.1, 3.0, 0.001
);
auto multi_end_jacobian = std::chrono::high_resolution_clock::now();

// 单状态串行计算雅可比矩阵
auto jacobian_raw = ilqr_node_list[0]->dynamics_jacobian(
    x_list.block(0,0,6,1),
    u_list.block(0,0,2,1)
);
auto signal_jacobian = std::chrono::high_resolution_clock::now();

// 计算时间差
std::chrono::duration<double> multi_jacobian =
    std::chrono::duration<double>(multi_end_jacobian - multi_start_jacobian);
std::chrono::duration<double> signal_jacobian_c =
    std::chrono::duration<double>(signal_jacobian - multi_end_jacobian);

// 输出性能对比
std::cout << "multi_jacobian " << multi_jacobian.count() << " seconds" << std::endl;
std::cout << "signal_jacobian_c " << signal_jacobian_c.count() << " seconds" << std::endl;
```

**目的**：对比并行雅可比计算和串行计算的速度差异。

**2. 并行线搜索性能**（第 99-114 行）

```cpp
double best_alpha = 1.0;
double best_cost = 0.0;

// 并行线搜索（同时测试多个步长）
auto multi_start = std::chrono::high_resolution_clock::now();
solver.ParallelLinearSearch(alpha, best_alpha, best_cost);
auto multi_end = std::chrono::high_resolution_clock::now();

std::chrono::duration<double> multi_count =
    std::chrono::duration<double>(multi_end - multi_start);

// 对比串行 Forward
std::chrono::duration<double> forward_count =
    std::chrono::duration<double>(forward_end - forward_start);

std::cout << "multi linear search " << multi_count.count() << " seconds" << std::endl;
std::cout << "forward " << forward_count.count() << " seconds" << std::endl;
```

**目的**：对比并行线搜索和标准向前传播的性能。

**3. 数值正确性验证**（第 92-96 行）

```cpp
std::cout << "x " << x_list.block(0,0,6,1) << "\n "
          << "u " << u_list.block(0,0,2,1) << std::endl;

for(int index = 0; index < 1; ++index) {
    std::cout << jacobian_raw.first << "\n"
              << jacobian.first.block(0,index * 6,6,6) << "\n" << std::endl;
}
```

**目的**：验证并行计算和串行计算的结果是否一致。

---

### test_new_al_ilqr.cc - 功能测试

#### 测试流程（第 64-141 行）

```cpp
// 第一次优化：盒式约束 + 矩形障碍物
BoxConstraints<6, 2> constraints_obj(...);
// 构建节点列表
for (int i = 0; i <= num_points; ++i) {
    ilqr_node_list.push_back(std::make_shared<NewBicycleNode<BoxConstraints<6, 2>>>(...));
}
NewALILQR<6,2> solver(ilqr_node_list, init_state, left_obs, right_obs);
solver.optimize(50, 100, 1e-3);  // 完整优化

// 输出第一次优化结果
for(int i = 0; i < num_points - 1; ++i) {
    std::cout << "u_result " << solver.get_u_list().col(i).transpose() << std::endl;
}

// 第二次优化：二次约束 + 圆形障碍物
QuadraticConstraints<6, 2, 5> quad_constrants(...);
// 重新构建节点列表
for (int i = 0; i <= num_points; ++i) {
    q_ilqr_node_list.push_back(std::make_shared<NewBicycleNode<QuadraticConstraints<6, 2, 5>>>(...));
}
NewALILQR<6,2> q_solver(q_ilqr_node_list, init_state);
q_solver.optimize(30, 100, 1e-3);  // 完整优化

// 输出第二次优化结果
for(int i = 0; i < num_points - 1; ++i) {
    std::cout << "q u_result " << q_solver.get_u_list().col(i).transpose() << std::endl;
}
```

**执行两次完整的 AL-iLQR 优化**，测试不同约束类型的求解能力。

---

## 参数差异分析

### 代价函数权重矩阵

#### test_new_al_ilqr_signal.cc（第 41-44 行）

```cpp
Q_fast.diagonal() << 1, 1, 1, 1, 1, 1;
Q_fast *= 1.0e16;  // 结果：diag([1e16, 1e16, 1e16, 1e16, 1e16, 1e16])
R_fast = Eigen::MatrixXd::Identity(2, 2) * 1.0;
```

**特点**：
- Q 权重**极大**（1e16），强制严格跟踪参考轨迹
- R 权重**极小**（1.0），几乎不惩罚控制输入
- **目的**：测试算法在极端参数下的数值稳定性

#### test_new_al_ilqr.cc（第 50-53 行）

```cpp
Q_fast.diagonal() << 1e-1, 1e-1, 1e-0, 1e-9, 1e-6, 1e-6;
Q_fast *= 1e3;  // 结果：diag([100, 100, 1000, 0.001, 1, 1])
R_fast = Eigen::MatrixXd::Identity(2, 2) * 1e2;
```

**特点**：
- Q 权重**适中**（100-1000），平衡跟踪精度
- R 权重**适中**（100），平衡控制平滑性
- **目的**：实际应用中的合理参数配置

### 权重对比表

| 维度 | signal 版本 Q | standard 版本 Q | 差异倍数 |
|------|---------------|-----------------|----------|
| x | 1e16 | 100 | **1e14 倍** |
| y | 1e16 | 100 | **1e14 倍** |
| θ | 1e16 | 1000 | **1e13 倍** |
| δ | 1e16 | 0.001 | **1e19 倍** |
| v | 1e16 | 1 | **1e16 倍** |
| a | 1e16 | 1 | **1e16 倍** |

| 控制 | signal 版本 R | standard 版本 R | 差异倍数 |
|------|---------------|-----------------|----------|
| δ̇ | 1.0 | 100 | **1/100** |
| ȧ | 1.0 | 100 | **1/100** |

**影响分析**：
- signal 版本：优化器会**不惜一切代价**跟踪参考轨迹，控制输入可能非常激进
- standard 版本：平衡跟踪精度和控制平滑性，结果更实用

---

## 输出内容对比

### test_new_al_ilqr_signal.cc 输出

```
x [状态向量第1个点]
u [控制向量第1个点]
[并行雅可比矩阵]
[串行雅可比矩阵]

multi linear search 0.00123 seconds
forward 0.00045 seconds
multi_jacobian 0.00089 seconds
signal_jacobian_c 0.00012 seconds
```

**关注点**：算法性能和数值正确性

### test_new_al_ilqr.cc 输出

```
u_result 0.0123 -0.0456  (第1个点)
u_result 0.0234 -0.0567  (第2个点)
...
x_result 0.0 0.0 0.0 0.0 10.0 0.0  (第1个点)
x_result 1.0 0.5 0.1 0.02 9.8 0.1  (第2个点)
...
q u_result [二次约束优化的控制序列]
q x_result [二次约束优化的状态序列]
optimize took 0.234 seconds
```

**关注点**：优化结果的完整轨迹

---

## 使用的特殊功能

### test_new_al_ilqr_signal.cc 独有功能

#### 1. 并行雅可比计算（第 86 行）

```cpp
auto jacobian = ParallelFullBicycleDynamicJacobianRK2<50>(
    x_list.block(0,0,6,50),
    u_list.block(0,0,2,50),
    0.1, 3.0, 0.001
);
```

这是一个模板函数，同时计算 50 个状态点的雅可比矩阵。

**来源**：`model/parallel_compution_function.h`

**优势**：
- 向量化计算（SIMD 优化）
- 减少函数调用开销
- 可能提速 5-10 倍

#### 2. 直接调用内部方法（第 71-73 行）

```cpp
solver.linearizedInitialGuess();
solver.CalcDerivatives();
solver.Backward();
solver.Forward();
```

**注意**：这些是内部方法，正常使用应该调用 `solver.optimize()`。

### test_new_al_ilqr.cc 独有功能

#### 1. 动态障碍物约束（第 78-87 行）

```cpp
Eigen::Matrix<double, 2, 4> left_car;
left_car << 32, 32, 28, 28,   // 矩形4个顶点的x坐标
            14, 16, 16, 14;    // 矩形4个顶点的y坐标
std::vector<Eigen::Matrix<double, 2, 4>> left_obs;
left_obs.push_back(left_car);

NewALILQR<6,2> solver(ilqr_node_list, init_state, left_obs, right_obs);
```

使用带障碍物的构造函数，触发**动态线性约束**的生成。

**原理**（参见 `new_al_ilqr.h:236-306`）：
- 在每次迭代中，检查车辆位置是否进入障碍物区域
- 动态添加线性约束 `Ax + Bu ≤ c` 阻止进入
- 约束仅在必要时激活（稀疏约束）

#### 2. 二次约束测试（第 99-133 行）

```cpp
QuadraticConstraints<6, 2, 5> quad_constrants(Q, A, B, C);
```

完整测试了二次约束的求解能力。

---

## 性能测试原理详解

### 为什么需要 signal 版本？

在算法开发中，性能优化是关键。`test_new_al_ilqr_signal.cc` 提供了：

1. **性能基准**：
   - 测量每个子模块的耗时
   - 识别性能瓶颈
   - 验证优化效果

2. **数值验证**：
   - 对比并行和串行算法的输出
   - 确保优化没有引入错误
   - 调试数值精度问题

3. **回归测试**：
   - 代码修改后，确保性能不退化
   - CI/CD 集成性能监控

### 典型性能测试结果示例

假设输出为：
```
multi linear search 0.00123 seconds
forward 0.00045 seconds
multi_jacobian 0.00089 seconds
signal_jacobian_c 0.00012 seconds
```

**分析**：
- 并行线搜索 (0.00123s) 比串行 Forward (0.00045s) **慢 2.7 倍**？
  - 可能原因：PARALLEL_NUM 太小，并行开销大于收益
  - 优化方向：增加 PARALLEL_NUM 或优化内存访问模式

- 并行雅可比 (0.00089s) vs 单个串行雅可比 (0.00012s)
  - 并行计算 50 个点耗时 0.00089s
  - 串行计算 1 个点耗时 0.00012s
  - 预计串行计算 50 个点需要：0.00012 × 50 = 0.006s
  - **加速比**: 0.006 / 0.00089 ≈ **6.7 倍**
  - 结论：并行优化有效！

---

## 适用场景

### 使用 test_new_al_ilqr_signal.cc 当你需要：

✅ 性能分析和优化
✅ 对比不同算法实现的速度
✅ 调试数值计算问题
✅ 验证并行化的正确性
✅ 建立性能基准数据
✅ CI/CD 性能回归测试

**示例**：
```bash
# 编译并运行性能测试
bazel build //:test_new_al_ilqr_signal -c opt
bazel run //:test_new_al_ilqr_signal

# 输出可用于自动化性能监控
./test_new_al_ilqr_signal | grep "seconds" > perf_log.txt
```

### 使用 test_new_al_ilqr.cc 当你需要：

✅ 验证算法功能完整性
✅ 测试不同约束类型的求解
✅ 生成示例轨迹数据
✅ 调试约束处理逻辑
✅ 回归测试（功能层面）

**示例**：
```bash
# 编译并运行功能测试
bazel build //:test_new_al_ilqr -c opt
bazel run //:test_new_al_ilqr > results.txt

# 结果可用于可视化或进一步分析
python analyze_results.py results.txt
```

---

## 总结

| 对比维度 | test_new_al_ilqr_signal.cc | test_new_al_ilqr.cc |
|----------|----------------------------|---------------------|
| **测试类型** | 性能基准测试 | 功能完整性测试 |
| **优化执行** | 不执行完整优化 | 执行 2 次完整优化 |
| **时间测量** | 详细的子模块计时 | 仅输出总优化时间 |
| **参数设置** | 极端参数（Q=1e16） | 实用参数（Q=100-1000） |
| **约束测试** | 仅盒式约束 | 盒式 + 二次 + 动态线性 |
| **特殊功能** | 并行雅可比、并行线搜索对比 | 矩形/圆形障碍物 |
| **输出内容** | 性能数据 | 完整轨迹 |
| **使用频率** | 开发调优阶段 | 日常测试验证 |
| **运行时间** | < 1 秒（单次迭代） | 数秒到数十秒（完整优化） |

**关系定位**：
- `test_new_al_ilqr_signal.cc`：**开发者工具**（性能调优）
- `test_new_al_ilqr.cc`：**用户工具**（功能演示）

两者互补，共同保证代码质量！
