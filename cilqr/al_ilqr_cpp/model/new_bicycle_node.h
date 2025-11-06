#ifndef NEW_BICYCLE_NODE_H_
#define NEW_BICYCLE_NODE_H_

#include "new_ilqr_node.h"
#include <Eigen/Dense>
#include <memory>
#include <cmath>

/**
 * @file new_bicycle_node.h
 * @brief 完整动力学自行车模型节点 - 用于车辆轨迹优化的核心模型
 *
 * 本文件实现了一个包含 6 维状态、2 维控制的完整自行车动力学模型,
 * 适用于 iLQR 轨迹优化算法。模型考虑了车辆的位置、航向、转向角、速度和加速度。
 */

/**
 * @brief 完整动力学自行车模型节点类
 *
 * NewBicycleNode 实现了一个用于车辆轨迹规划的完整动力学模型。
 * 该模型继承自 NewILQRNode<6, 2>,其中:
 * - 状态维度: 6 (x, y, θ, δ, v, a)
 * - 控制维度: 2 (δ_rate, a_rate)
 *
 * ## 状态变量 (6 维)
 *
 * ```
 * state = [x, y, θ, δ, v, a]ᵀ
 * ```
 *
 * - `x` (state[0]): 车辆位置 x 坐标 (m)
 * - `y` (state[1]): 车辆位置 y 坐标 (m)
 * - `θ` (state[2]): 航向角 (rad), 车辆行驶方向相对于 x 轴的角度
 * - `δ` (state[3]): 前轮转向角 (rad), 前轮相对于车身纵轴的角度
 * - `v` (state[4]): 纵向速度 (m/s), 车辆沿车身方向的速度
 * - `a` (state[5]): 纵向加速度 (m/s²), 车辆沿车身方向的加速度
 *
 * ## 控制变量 (2 维)
 *
 * ```
 * control = [δ_rate, a_rate]ᵀ
 * ```
 *
 * - `δ_rate` (control[0]): 转向角速度 (rad/s), 前轮转角的变化率
 * - `a_rate` (control[1]): 加加速度 (jerk) (m/s³), 加速度的变化率
 *
 * ## 连续时间动力学方程
 *
 * ```
 * ẋ = v · cos(θ)
 * ẏ = v · sin(θ)
 * θ̇ = v · tan(δ) / [L · (1 + k·v²)]
 * δ̇ = δ_rate
 * v̇ = a
 * ȧ = a_rate
 * ```
 *
 * 其中:
 * - `L`: 车辆轴距 (前后轮轴之间的距离, 单位: m)
 * - `k`: 速度修正系数 (用于高速时的动力学修正, 通常很小如 0.001)
 *
 * ## 离散化方法: RK2 (二阶 Runge-Kutta)
 *
 * 为了在离散时间步长 dt 上精确求解连续动力学,使用 RK2 方法:
 *
 * ```
 * k₁ = f(xₖ, uₖ)
 * k₂ = f(xₖ + 0.5·dt·k₁, uₖ)
 * xₖ₊₁ = xₖ + dt·k₂
 * ```
 *
 * RK2 相比简单的欧拉法提供了更高的精度 (O(dt²) vs O(dt))。
 *
 * ## 角度归一化
 *
 * 由于 θ 和 δ 是角度变量,具有周期性 (2π),需要进行归一化处理:
 * - 将角度限制在 [-π, π] 范围内
 * - 避免角度跳变导致的数值问题
 *
 * ## 代价函数
 *
 * 代价函数由两部分组成:
 *
 * ```
 * cost = (x - x_goal)ᵀ Q (x - x_goal) + uᵀ R u + augmented_lagrangian_cost
 * ```
 *
 * 其中:
 * - `Q`: 状态跟踪权重矩阵 (6×6 对角矩阵)
 * - `R`: 控制平滑权重矩阵 (2×2 对角矩阵)
 * - `augmented_lagrangian_cost`: 约束惩罚项 (来自增广拉格朗日方法)
 *
 * ## 约束处理
 *
 * 约束通过模板参数 ConstraintsType 传入,支持:
 * - BoxConstraints: 盒式约束 (状态和控制的上下界)
 * - QuadraticConstraints: 二次约束 (如圆形障碍物避障)
 * - LinearConstraints: 线性约束
 *
 * ## 并行计算支持
 *
 * 提供了并行版本的动力学和代价函数计算:
 * - parallel_dynamics(): 同时评估多个状态-控制对的动力学
 * - parallel_cost(): 同时评估多个状态-控制对的代价
 * - 用于并行线搜索,显著提高优化效率
 *
 * ## 典型应用场景
 *
 * - 自动驾驶车辆的轨迹规划
 * - 车辆避障和路径跟踪
 * - 停车场景的轨迹优化
 * - 车辆动力学仿真
 *
 * @tparam ConstraintsType 约束类型 (如 BoxConstraints<6,2>)
 *
 * @note 此模型适用于低速到中速场景 (v < 20 m/s)
 * @note 对于高速场景,可能需要考虑更复杂的车辆动力学 (如轮胎侧偏力)
 *
 * @see NewILQRNode 基类,定义了 iLQR 节点的接口
 * @see BoxConstraints 盒式约束实现
 * @see QuadraticConstraints 二次约束实现
 */
template <class ConstraintsType>
class NewBicycleNode : public NewILQRNode<6, 2> {
public:
    // 类型别名,简化代码
    using VectorState = typename NewILQRNode<6, 2>::VectorState;      ///< 状态向量 (6×1)
    using VectorControl = typename NewILQRNode<6, 2>::VectorControl;  ///< 控制向量 (2×1)
    using MatrixA = typename NewILQRNode<6, 2>::MatrixA;              ///< 动力学雅可比 A (6×6)
    using MatrixB = typename NewILQRNode<6, 2>::MatrixB;              ///< 动力学雅可比 B (6×2)
    using MatrixQ = typename NewILQRNode<6, 2>::MatrixQ;              ///< 状态代价权重矩阵 (6×6)
    using MatrixR = typename NewILQRNode<6, 2>::MatrixR;              ///< 控制代价权重矩阵 (2×2)

    /**
     * @brief 构造自行车模型节点
     *
     * @param L 车辆轴距 (m), 前后轮轴之间的距离, 典型值: 2.5~3.5m (轿车)
     * @param dt 时间步长 (s), 离散化时间间隔, 典型值: 0.05~0.2s
     * @param k 速度修正系数 (无量纲), 用于高速动力学修正, 典型值: 0.001~0.01
     *          该系数用于修正高速时的转向率: θ̇ = v·tan(δ) / [L·(1 + k·v²)]
     * @param goal 目标状态 (6×1), 当前时刻的参考轨迹点
     * @param Q 状态代价权重矩阵 (6×6), 对角矩阵, 定义各状态维度的跟踪重要性
     * @param R 控制代价权重矩阵 (2×2), 对角矩阵, 定义控制平滑性的重要性
     * @param constraints 约束对象, 定义状态和控制的约束条件
     *
     * @note k 值越大,高速时的转向率越小 (更符合实际车辆特性)
     * @note 对于低速场景 (v < 5 m/s), k 的影响很小,可以设为 0.001
     *
     * @example
     * ```cpp
     * // 创建带盒式约束的自行车模型节点
     * double L = 3.0;    // 轴距 3m
     * double dt = 0.1;   // 时间步长 0.1s
     * double k = 0.001;  // 速度修正系数
     *
     * Eigen::Vector<double, 6> goal;
     * goal << 10.0, 5.0, 0.5, 0.0, 10.0, 0.0;  // 目标状态
     *
     * Eigen::Matrix<double, 6, 6> Q = Eigen::Matrix<double, 6, 6>::Identity();
     * Q.diagonal() << 100, 100, 1000, 0.001, 1, 1;  // 位置和航向权重高
     *
     * Eigen::Matrix<double, 2, 2> R = Eigen::Matrix<double, 2, 2>::Identity() * 100;
     *
     * BoxConstraints<6, 2> constraints(state_min, state_max, control_min, control_max);
     *
     * NewBicycleNode<BoxConstraints<6, 2>> node(L, dt, k, goal, Q, R, constraints);
     * ```
     */
    NewBicycleNode(double L, double dt, double k, const VectorState& goal, const MatrixQ& Q, const MatrixR& R, const ConstraintsType& constraints)
        : NewILQRNode<6, 2>(goal), constraints_(constraints), L_(L), dt_(dt), k_(k), Q_(Q), R_(R) {}

    /**
     * @brief 归一化状态向量中的角度变量
     *
     * 将状态向量中的角度变量 (θ 和 δ) 归一化到 [-π, π] 范围内。
     * 这对于避免角度跳变和数值不稳定问题至关重要。
     *
     * @param state 状态向量 (6×1), 会被原地修改
     *
     * @note 只归一化 state[2] (θ) 和 state[3] (δ)
     * @note 其他状态变量 (x, y, v, a) 不需要归一化
     *
     * @example
     * ```cpp
     * VectorState state;
     * state << 10.0, 5.0, 7.0, 0.5, 10.0, 0.0;  // θ = 7.0 > π
     * normalize_state(state);
     * // state[2] 现在是 7.0 - 2π ≈ 0.717
     * ```
     */
    void normalize_state(VectorState& state) const {
        this->normalize_angle(state(2)); // 归一化 θ (航向角)
        this->normalize_angle(state(3)); // 归一化 δ (前轮转角)
    }

    /**
     * @brief 计算离散时间动力学 (状态转移)
     *
     * 使用 RK2 (二阶 Runge-Kutta) 方法对连续时间动力学进行离散化,
     * 计算下一时刻的状态: xₖ₊₁ = f(xₖ, uₖ)
     *
     * RK2 算法:
     * ```
     * k₁ = f_continuous(xₖ, uₖ)
     * x_mid = xₖ + 0.5·dt·k₁
     * k₂ = f_continuous(x_mid, uₖ)
     * xₖ₊₁ = xₖ + dt·k₂
     * ```
     *
     * @param state 当前状态 xₖ (6×1)
     * @param control 当前控制 uₖ (2×1)
     * @return 下一时刻状态 xₖ₊₁ (6×1), 角度已归一化
     *
     * @note override 关键字表示覆盖基类 NewILQRNode 的虚函数
     * @note RK2 的时间复杂度为 O(dt²), 比欧拉法 O(dt) 更精确
     *
     * @example
     * ```cpp
     * VectorState x;
     * x << 0.0, 0.0, 0.0, 0.0, 10.0, 0.0;  // 初始状态
     * VectorControl u;
     * u << 0.1, 0.0;  // 转向角速度 0.1 rad/s
     *
     * VectorState x_next = node.dynamics(x, u);
     * // x_next[0] ≈ 0.0 + dt·10·cos(0) = 1.0 (如果 dt=0.1)
     * ```
     */
    VectorState dynamics(const Eigen::Ref<const VectorState>& state, const Eigen::Ref<const VectorControl>& control) const override {
        VectorState state_next;

        // RK2 步骤 1: 计算初始斜率 k₁
        VectorState k1 = dynamics_continuous(state, control);
        // RK2 步骤 2: 计算中点状态
        VectorState mid_state = state + 0.5 * dt_ * k1;
        // RK2 步骤 3: 在中点处计算斜率 k₂
        VectorState k2 = dynamics_continuous(mid_state, control);

        // RK2 步骤 4: 使用中点斜率更新状态
        state_next = state + dt_ * k2;
        // 归一化角度变量
        normalize_state(state_next);

        return state_next;
    }

    /**
     * @brief 批量计算连续时间动力学 (向量化版本)
     *
     * 这是 dynamics_continuous 的向量化版本,用于并行线搜索中同时计算多个状态-控制对的动力学。
     *
     * 功能: 给定 N 个状态和 N 个控制输入,同时计算所有的状态导数。
     *
     * @param state 状态矩阵 (6 × N), 每列是一个状态向量
     *                第0行: x 坐标列表
     *                第1行: y 坐标列表
     *                第2行: θ 航向角列表
     *                第3行: δ 前轮转角列表
     *                第4行: v 速度列表
     *                第5行: a 加速度列表
     * @param control 控制矩阵 (2 × N), 每列是一个控制向量
     *                第0行: δ_rate 转向角速度列表
     *                第1行: a_rate 加加速度列表
     * @return 状态导数矩阵 (6 × N), 每列是对应状态的导数
     *
     * ## 向量化实现技巧
     *
     * 使用 Eigen 的数组运算 (.array()) 进行逐元素操作:
     * - v * cos(θ): 通过 v_array * theta_array.cos() 一次性计算所有 N 个状态
     * - v * sin(θ): 类似地批量计算
     * - v * tan(δ) / [L(1+k*v²)]: 批量计算转向率
     *
     * ## 性能优势
     *
     * 相比循环调用 dynamics_continuous:
     * - 利用 CPU 的 SIMD 指令集 (向量化)
     * - 减少函数调用开销
     * - 通常快 5-10 倍 (取决于 N 的大小)
     *
     * @note 用于 ParallelLinearSearch 中同时评估多个线搜索步长
     * @note N 的典型值由 PARALLEL_NUM 宏定义 (通常是 8-16)
     *
     * @see parallel_dynamics() 使用 RK2 对此函数进行离散化
     * @see ParallelLinearSearch() 并行线搜索的主函数
     */
    Eigen::MatrixXd parallel_dynamics_continuous(const Eigen::Ref<const Eigen::MatrixXd>& state, const Eigen::Ref<const Eigen::MatrixXd>& control) const {
        int state_rows = state.rows();  // 应该是 6
        int state_cols = state.cols();  // 并行数量 N

        // 提取各状态变量的向量 (1 × N 行向量)
        auto theta_list_matrix_raw = state.row(2);  // 航向角列表
        auto delta_list_matrix_raw = state.row(3);  // 前轮转角列表
        Eigen::MatrixXd v_list_matrix_raw = state.row(4);  // 速度列表
        auto a_list_matrix_raw = state.row(5);  // 加速度列表

        // 批量计算位置导数: ẋ = v*cos(θ), ẏ = v*sin(θ)
        auto v_cos_theta_array = v_list_matrix_raw.array() * theta_list_matrix_raw.array().cos();
        auto v_sin_theta_array = v_list_matrix_raw.array() * theta_list_matrix_raw.array().sin();

        // 创建全1矩阵,用于计算分母: 1 + k*v²
        Eigen::MatrixXd ones = v_list_matrix_raw;
        ones.setOnes();

        // 批量计算航向角导数: θ̇ = v*tan(δ) / [L*(1 + k*v²)]
        auto v_tan_delta_array_divide_L = v_list_matrix_raw.array()
                                          * delta_list_matrix_raw.array().tan()
                                          * (ones.array() + k_ * v_list_matrix_raw.array() * v_list_matrix_raw.array()).inverse()
                                          / L_;

        // 组装状态导数矩阵
        Eigen::MatrixXd state_dot(state_rows, state_cols);

        state_dot.row(0) = v_cos_theta_array.matrix();        // ẋ = v*cos(θ)
        state_dot.row(1) = v_sin_theta_array.matrix();        // ẏ = v*sin(θ)
        state_dot.row(2) = v_tan_delta_array_divide_L.matrix();  // θ̇ = v*tan(δ)/[L*(1+k*v²)]
        state_dot.row(3) = control.row(0);                    // δ̇ = δ_rate
        state_dot.row(4) = a_list_matrix_raw;                 // v̇ = a
        state_dot.row(5) = control.row(1);                    // ȧ = a_rate

        return state_dot;
    }


    /**
     * @brief 批量计算离散时间动力学 (RK2向量化版本)
     *
     * 这是 dynamics 的向量化版本,同时对 N 个状态-控制对执行 RK2 积分。
     *
     * RK2 算法 (批量版本):
     * ```
     * k₁ = f_continuous(state, control)           # 6×N 矩阵
     * mid_state = state + 0.5·dt·k₁                # 6×N 矩阵
     * k₂ = f_continuous(mid_state, control)        # 6×N 矩阵
     * next_state = state + dt·k₂                   # 6×N 矩阵
     * ```
     *
     * @param state 状态矩阵 (6 × N), 每列是一个当前状态
     * @param control 控制矩阵 (2 × N), 每列是一个当前控制
     * @return 下一时刻状态矩阵 (6 × N), 每列是对应的下一状态
     *
     * ## 应用场景
     *
     * 在并行线搜索中,需要同时评估多个步长 α:
     * - α₁ = 1.0, α₂ = 1/3, α₃ = 1/9, ...
     * - 对每个 αᵢ,计算对应的轨迹和总代价
     * - 选择代价最小的 αᵢ
     *
     * 使用此函数可以一次性前向仿真所有候选步长的轨迹。
     *
     * @note override 关键字表示覆盖基类的虚函数
     * @note 此函数不进行角度归一化 (假设输入已归一化)
     * @note 性能提升显著,是并行线搜索高效的关键
     *
     * @see parallel_dynamics_continuous() 批量连续动力学计算
     * @see ParallelLinearSearch() 使用此函数的并行线搜索
     *
     * @example
     * ```cpp
     * // 同时评估 8 个不同步长的轨迹
     * Eigen::Matrix<double, 6, 8> states;  // 8个状态
     * Eigen::Matrix<double, 2, 8> controls;  // 8个控制
     * // ... 初始化 states 和 controls ...
     *
     * Eigen::Matrix<double, 6, 8> next_states = node.parallel_dynamics(states, controls);
     * // next_states 的每一列是对应输入的下一状态
     * ```
     */
    Eigen::MatrixXd parallel_dynamics(const Eigen::Ref<const Eigen::MatrixXd>& state, const Eigen::Ref<const Eigen::MatrixXd>& control) const override {
        // RK2 步骤 1: 计算初始斜率
        Eigen::MatrixXd state_dot = parallel_dynamics_continuous(state, control);
        // RK2 步骤 2: 计算中点状态
        Eigen::MatrixXd mid_state = state + state_dot * dt_ * 0.5;
        // RK2 步骤 3: 在中点处计算斜率
        Eigen::MatrixXd mid_state_dot = parallel_dynamics_continuous(mid_state, control);
        // RK2 步骤 4: 使用中点斜率更新状态
        Eigen::MatrixXd next_state = state + mid_state_dot * dt_;
        return next_state;
    }



    /**
     * @brief 计算连续时间动力学 (状态导数)
     *
     * 计算自行车模型的连续时间动力学方程: ẋ = f(x, u)
     *
     * 动力学方程:
     * ```
     * ẋ = v · cos(θ)                      位置 x 方向速度
     * ẏ = v · sin(θ)                      位置 y 方向速度
     * θ̇ = v · tan(δ) / [L·(1 + k·v²)]    航向角变化率 (考虑速度修正)
     * δ̇ = δ_rate                          前轮转角变化率
     * v̇ = a                               纵向加速度
     * ȧ = a_rate                          加加速度 (jerk)
     * ```
     *
     * ## 几何解释
     *
     * - `ẋ, ẏ`: 车辆质心在全局坐标系中的速度分量
     * - `θ̇`: 根据自行车模型几何关系,转向率与速度、转角和轴距相关
     * - `k·v²` 项: 高速修正,模拟实际车辆在高速时转向能力下降的特性
     *
     * @param state 当前状态 (6×1)
     * @param control 当前控制 (2×1)
     * @return 状态导数 ẋ (6×1)
     *
     * @note 角度会自动归一化到 [-π, π]
     * @note 当 k=0 时,退化为标准自行车模型: θ̇ = v·tan(δ)/L
     *
     * @example
     * ```cpp
     * VectorState x;
     * x << 0, 0, 0, 0.1, 10, 0;  // θ=0, δ=0.1 rad, v=10 m/s
     * VectorControl u;
     * u << 0, 0;
     *
     * VectorState x_dot = node.dynamics_continuous(x, u);
     * // x_dot[0] ≈ 10·cos(0) = 10 m/s
     * // x_dot[1] ≈ 10·sin(0) = 0 m/s
     * // x_dot[2] ≈ 10·tan(0.1) / 3 ≈ 0.335 rad/s (假设 L=3m, k=0)
     * ```
     */
    VectorState dynamics_continuous(const Eigen::Ref<const VectorState>& state, const Eigen::Ref<const VectorControl>& control) const {
        VectorState x_dot;

        // 提取状态变量
        double theta = state(2);    // 航向角
        double delta = state(3);    // 前轮转角
        double v = state(4);        // 纵向速度
        double a = state(5);        // 纵向加速度
        double u1 = control(0);     // 转向角速度
        double u2 = control(1);     // 加加速度

        // 归一化角度到 [-π, π]
        this->normalize_angle(theta);
        this->normalize_angle(delta);

        // 计算状态导数 (连续时间动力学方程)
        x_dot(0) = v * std::cos(theta);                             // ẋ
        x_dot(1) = v * std::sin(theta);                             // ẏ
        x_dot(2) = v * std::tan(delta) / (L_ * (1 + k_ * v * v));  // θ̇ (带速度修正)
        x_dot(3) = u1;                                              // δ̇
        x_dot(4) = a;                                               // v̇
        x_dot(5) = u2;                                              // ȧ

        return x_dot;
    }


    /**
     * @brief 计算动力学函数的雅可比矩阵
     *
     * 计算离散时间动力学 xₖ₊₁ = f(xₖ, uₖ) 对状态和控制的偏导数。
     * 这些雅可比矩阵用于 iLQR 的 Backward Pass。
     *
     * 返回值:
     * ```
     * Jx = ∂f/∂x  (6×6 矩阵)  状态雅可比矩阵
     * Ju = ∂f/∂u  (6×2 矩阵)  控制雅可比矩阵
     * ```
     *
     * ## 雅可比矩阵的作用
     *
     * 在 iLQR 算法的 Backward Pass 中,雅可比矩阵用于计算 Q 函数:
     * ```
     * Qₓ = lₓ + fₓᵀ Vₓ'
     * Qᵤ = lᵤ + fᵤᵀ Vₓ'
     * ```
     *
     * ## 实现细节
     *
     * 由于使用了 RK2 离散化,雅可比矩阵不是简单的 I + dt·∂f_cont/∂x,
     * 而是通过链式法则计算 RK2 过程的完整导数。
     *
     * @param state 当前状态 (6×1)
     * @param control 当前控制 (2×1)
     * @return std::pair<Jx, Ju>
     *         - Jx: 状态雅可比矩阵 ∂f/∂x (6×6)
     *         - Ju: 控制雅可比矩阵 ∂f/∂u (6×2)
     *
     * @note 雅可比矩阵的计算考虑了 RK2 积分过程
     * @note 角度变量在计算前会自动归一化
     *
     * @warning 雅可比矩阵的解析形式较复杂,由符号计算工具生成
     *
     * @example
     * ```cpp
     * auto [Jx, Ju] = node.dynamics_jacobian(x, u);
     * // Jx(0, 4) 是 ∂x_{k+1}[0] / ∂x_k[4], 即下一时刻 x 位置对当前速度的导数
     * // Ju(2, 0) 是 ∂x_{k+1}[2] / ∂u_k[0], 即下一时刻航向角对转向角速度的导数
     * ```
     */
    std::pair<MatrixA, MatrixB> dynamics_jacobian(const Eigen::Ref<const VectorState>& state, const Eigen::Ref<const VectorControl>& control) const override {
        double theta = state[2], delta = state[3], v = state[4], a = state[5];
        double u1 = control[0];

        this->normalize_angle(theta);
        this->normalize_angle(delta);

        double dt = dt_;
        double L = L_;
        double k = k_;

        // double theta_mid = theta + 0.5 * dt * v * std::tan(delta) / (L * (k * v * v + 1));
        double v_term = 0.5 * a * dt + v;
        // double tan_delta = std::tan(delta);
        double tan_delta_mid = std::tan(delta + 0.5 * dt * u1);
        // double k_v_sq = k * v * v;
        double k_v_mid_sq = k * v_term * v_term;
        // double denom = L * (k_v_sq + 1);
        double denom_mid = L * (k_v_mid_sq + 1);
        // double cos_theta_mid = std::cos(theta_mid);
        // double sin_theta_mid = std::sin(theta_mid);

        

        // Define Jx
        Eigen::MatrixXd Jx(6, 6);
        Jx << 1, 0, -dt * (0.5 * a * dt + v) * sin(theta + 0.5 * dt * v * tan(delta) / (L * (k * v * v + 1))),
          -0.5 * dt * dt * v * (0.5 * a * dt + v) * (tan(delta) * tan(delta) + 1) * sin(theta + 0.5 * dt * v * tan(delta) / (L * (k * v * v + 1))) / (L * (k * v * v + 1)),
          -dt * (0.5 * a * dt + v) * (-1.0 * dt * k * v * v * tan(delta) / (L * (k * v * v + 1) * (k * v * v + 1)) + 0.5 * dt * tan(delta) / (L * (k * v * v + 1))) * sin(theta + 0.5 * dt * v * tan(delta) / (L * (k * v * v + 1))) + dt * cos(theta + 0.5 * dt * v * tan(delta) / (L * (k * v * v + 1))),
          0.5 * dt * dt * cos(theta + 0.5 * dt * v * tan(delta) / (L * (k * v * v + 1))),
          0, 1, dt * (0.5 * a * dt + v) * cos(theta + 0.5 * dt * v * tan(delta) / (L * (k * v * v + 1))),
          0.5 * dt * dt * v * (0.5 * a * dt + v) * (tan(delta) * tan(delta) + 1) * cos(theta + 0.5 * dt * v * tan(delta) / (L * (k * v * v + 1))) / (L * (k * v * v + 1)),
          dt * (0.5 * a * dt + v) * (-1.0 * dt * k * v * v * tan(delta) / (L * (k * v * v + 1) * (k * v * v + 1)) + 0.5 * dt * tan(delta) / (L * (k * v * v + 1))) * cos(theta + 0.5 * dt * v * tan(delta) / (L * (k * v * v + 1))) + dt * sin(theta + 0.5 * dt * v * tan(delta) / (L * (k * v * v + 1))),
          0.5 * dt * dt * sin(theta + 0.5 * dt * v * tan(delta) / (L * (k * v * v + 1))),
          0, 0, 1, dt * (0.5 * a * dt + v) * (tan(delta + 0.5 * dt * u1) * tan(delta + 0.5 * dt * u1) + 1) / (L * (k * (0.5 * a * dt + v) * (0.5 * a * dt + v) + 1)),
          -dt * k * (0.5 * a * dt + v) * (1.0 * a * dt + 2 * v) * tan(delta + 0.5 * dt * u1) / (L * (k * (0.5 * a * dt + v) * (0.5 * a * dt + v) + 1) * (k * (0.5 * a * dt + v) * (0.5 * a * dt + v) + 1)) + dt * tan(delta + 0.5 * dt * u1) / (L * (k * (0.5 * a * dt + v) * (0.5 * a * dt + v) + 1)),
          -1.0 * dt * dt * k * (0.5 * a * dt + v) * (0.5 * a * dt + v) * tan(delta + 0.5 * dt * u1) / (L * (k * (0.5 * a * dt + v) * (0.5 * a * dt + v) + 1) * (k * (0.5 * a * dt + v) * (0.5 * a * dt + v) + 1)) + 0.5 * dt * dt * tan(delta + 0.5 * dt * u1) / (L * (k * (0.5 * a * dt + v) * (0.5 * a * dt + v) + 1)),
          0, 0, 0, 1, 0, 0,
          0, 0, 0, 0, 1, dt,
          0, 0, 0, 0, 0, 1;


        // Define Ju
        Eigen::MatrixXd Ju(6, 2);
        Ju << 
            0, 0,
            0, 0,
            0.5 * dt * dt * v_term * (tan_delta_mid * tan_delta_mid + 1) / denom_mid, 0,
            dt, 0,
            0, 0.5 * dt * dt,
            0, dt;

        return {Jx, Ju};
    }

    /**
     * @brief 计算动力学函数的 Hessian 张量 (二阶偏导数)
     *
     * 计算离散动力学函数对状态的二阶偏导数: ∂²f/∂x²
     *
     * ## Hessian 张量的结构
     *
     * 对于 6 维状态和 6 维输出的动力学,Hessian 是一个 6×6×6 的三阶张量。
     * 为了简化表示和计算,只返回前 3 个输出分量的 Hessian:
     *
     * ```
     * H₀ = ∂²f₀/∂x² (6×6)  对应 ẋ (x 位置的导数)
     * H₁ = ∂²f₁/∂x² (6×6)  对应 ẏ (y 位置的导数)
     * H₂ = ∂²f₂/∂x² (6×6)  对应 θ̇ (航向角的导数)
     * ```
     *
     * 其中 Hᵢ(j,k) = ∂²fᵢ/∂xⱼ∂xₖ
     *
     * ## 在 iLQR 中的作用
     *
     * Hessian 用于 Backward Pass 中更精确的二阶近似:
     * ```
     * Qₓₓ = lₓₓ + fₓᵀ Vₓₓ fₓ + Σᵢ Vₓ[i] * ∂²fᵢ/∂x²
     *                           \___________________/
     *                              Hessian 修正项
     * ```
     *
     * 这个二阶修正项提高了非线性系统的近似精度,使 iLQR 收敛更快更稳定。
     *
     * ## 非零元素说明
     *
     * 由于自行车模型的特殊结构,大部分 Hessian 元素为 0。
     * 主要的非零元素来自三角函数的二阶导数:
     *
     * - `H₀(θ, θ)`: ∂²ẋ/∂θ² = -v*cos(θ)  (ẋ = v*cos(θ) 的二阶导)
     * - `H₁(θ, θ)`: ∂²ẏ/∂θ² = -v*sin(θ)  (ẏ = v*sin(θ) 的二阶导)
     * - `H₂(δ, δ)`, `H₂(δ, v)`, `H₂(v, v)`: θ̇ = v*tan(δ)/[L*(1+k*v²)] 的二阶导
     *
     * @param state 当前状态 (6×1)
     * @param control 当前控制 (2×1)
     * @return std::tuple<H₀, H₁, H₂>
     *         - H₀: ∂²f₀/∂x² (6×6), 对应 ẋ 的 Hessian
     *         - H₁: ∂²f₁/∂x² (6×6), 对应 ẏ 的 Hessian
     *         - H₂: ∂²f₂/∂x² (6×6), 对应 θ̇ 的 Hessian
     *
     * @note 后三个状态变量 (δ, v, a) 的动力学是线性的,因此 Hessian 为 0
     * @note Hessian 矩阵是对称的: Hᵢ(j,k) = Hᵢ(k,j)
     * @note 此函数在 Backward Pass 中每个时间步调用一次
     *
     * @warning Hessian 的解析形式非常复杂,通常由符号计算工具 (如 Mathematica) 生成
     *
     * @example
     * ```cpp
     * auto [H0, H1, H2] = node.dynamics_hessian_fxx(x, u);
     * // H0(2, 2) 是 ∂²ẋ/∂θ²
     * // H2(3, 4) 是 ∂²θ̇/∂δ∂v
     * ```
     */
    std::tuple<MatrixA, MatrixA, MatrixA> dynamics_hessian_fxx(const Eigen::Ref<const VectorState>& state, const Eigen::Ref<const VectorControl>& control) const override {
        MatrixA ans1, ans2, ans3;  // 分别对应 ∂²f₀/∂x², ∂²f₁/∂x², ∂²f₂/∂x²
        ans1.setZero();
        ans2.setZero();
        ans3.setZero();

        // 提取状态变量
        double theta = state[2], delta = state[3], v = state[4];

        // 归一化角度
        this->normalize_angle(theta);
        this->normalize_angle(delta);

        double dt = dt_;
        double L = L_;
        double k = k_;

        // 预计算常用的三角函数和复合变量
        double sin_theta = std::sin(theta);
        double cos_theta = std::cos(theta);
        double tan_delta = std::tan(delta);
        double tan_delta_square_plus_one = tan_delta * tan_delta + 1;  // sec²(δ)
        double k_v_square = k * v * v;
        double k_v_square_plus_one = k_v_square + 1;

        // ===== ans1: ∂²f₀/∂x² (ẋ = v*cos(θ) 的 Hessian) =====
        ans1(2, 2) = -dt * v * cos_theta;  // ∂²ẋ/∂θ² = -v*cos(θ)
        ans1(2, 4) = -dt * sin_theta;       // ∂²ẋ/∂θ∂v = -sin(θ)
        ans1(4, 2) = ans1(2, 4);            // 对称性: ∂²ẋ/∂v∂θ = ∂²ẋ/∂θ∂v

        // ===== ans2: ∂²f₁/∂x² (ẏ = v*sin(θ) 的 Hessian) =====
        ans2(2, 2) = -dt * v * sin_theta;  // ∂²ẏ/∂θ² = -v*sin(θ)
        ans2(2, 4) = dt * cos_theta;        // ∂²ẏ/∂θ∂v = cos(θ)
        ans2(4, 2) = ans2(2, 4);            // 对称性

        // ===== ans3: ∂²f₂/∂x² (θ̇ = v*tan(δ)/[L*(1+k*v²)] 的 Hessian) =====
        // ∂²θ̇/∂δ² (转角的二阶效应)
        ans3(3, 3) = 2 * dt * v * tan_delta_square_plus_one * tan_delta / (k_v_square_plus_one * L);
        // ∂²θ̇/∂δ∂v (转角与速度的交叉项)
        ans3(3, 4) = dt * (1 - k * v * v) * tan_delta_square_plus_one / (k_v_square_plus_one * L) / (k_v_square_plus_one);
        ans3(4, 3) = ans3(3, 4);  // 对称性
        // ∂²θ̇/∂v² (速度的二阶效应,考虑 k*v² 修正项)
        ans3(4, 4) = dt * 2 * k * v * (k_v_square - 3) * tan_delta / L / k_v_square_plus_one / k_v_square_plus_one / k_v_square_plus_one;

        return {ans1, ans2, ans3};
    }

    /**
     * @brief 计算阶段代价函数
     *
     * 计算单个时间步的总代价,包括状态跟踪代价、控制平滑代价和约束惩罚代价。
     *
     * 代价函数公式:
     * ```
     * cost = (x - x_goal)ᵀ Q (x - x_goal)  +  uᵀ R u  +  L_aug(x, u, λ, μ)
     *        \_________________________/     \______/     \_______________/
     *           状态跟踪代价                控制代价       增广拉格朗日代价
     * ```
     *
     * 其中:
     * - `x - x_goal`: 状态误差向量 (6×1)
     * - `Q`: 状态权重矩阵 (6×6 对角矩阵), Q_ii 越大表示第 i 维状态越重要
     * - `R`: 控制权重矩阵 (2×2 对角矩阵), R_ii 越大表示惩罚第 i 维控制越重
     * - `L_aug`: 增广拉格朗日项,用于处理约束
     *
     * ## 代价函数的物理意义
     *
     * - **状态跟踪代价**: 惩罚偏离参考轨迹的程度
     *   - 如果 Q[0,0] 很大,优化器会努力跟踪 x 位置
     *   - 如果 Q[2,2] 很大,优化器会努力保持正确的航向角
     *
     * - **控制代价**: 惩罚控制输入的大小,保证平滑性和舒适性
     *   - R[0,0] 大 → 转向更平滑 (小的 δ_rate)
     *   - R[1,1] 大 → 加速更平滑 (小的 a_rate, 提高乘坐舒适性)
     *
     * - **约束代价**: 通过增广拉格朗日方法处理约束违反
     *   - 约束满足时,此项为 0
     *   - 约束违反时,此项会增加以引导优化器回到可行域
     *
     * @param state 当前状态 (6×1)
     * @param control 当前控制 (2×1)
     * @return 标量代价值
     *
     * @note 此函数对应 iLQR 中的阶段代价 l(x, u)
     * @note Q 和 R 通常是对角矩阵,简化计算
     *
     * @example
     * ```cpp
     * VectorState x;
     * x << 10.5, 5.2, 0.1, 0.0, 10.0, 0.0;
     * VectorControl u;
     * u << 0.05, 0.0;
     *
     * double c = node.cost(x, u);
     * // c = 状态误差加权平方和 + 控制加权平方和 + 约束惩罚
     * ```
     */
    double cost(const Eigen::Ref<const VectorState>& state, const Eigen::Ref<const VectorControl>& control) override {
        // 计算状态误差
        VectorState state_error = state - this->goal_;

        // 转换为数组格式以进行逐元素操作
        Eigen::Array<double, 6, 1> Q_array = (Q_.diagonal()).array();
        Eigen::Array<double, 6, 1> error_array = state_error.array();
        Eigen::Array<double, 2, 1> R_array = (R_.diagonal()).array();
        Eigen::Array<double, 2, 1> control_array = control.array();

        // 计算加权误差: Q_ii * error_i
        Eigen::Matrix<double, 6, 1> new_error = (error_array * Q_array).matrix();
        Eigen::Matrix<double, 2, 1> new_control = (R_array * control_array).matrix();

        // 状态代价: eᵀQe (通过先计算 Qe 再点乘 e 实现)
        double state_cost = (new_error.transpose() * state_error).value();
        // 控制代价: uᵀRu
        double control_cost = (new_control.transpose() * control).value();
        // 约束惩罚代价
        double constraints_cost = constraints_.augmented_lagrangian_cost(state, control);

        return state_cost + control_cost + constraints_cost;
    }

    /**
     * @brief 批量计算代价函数 (向量化版本)
     *
     * 同时计算 N 个状态-控制对的代价,用于并行线搜索。
     *
     * 对每一列 i,计算:
     * ```
     * cost[i] = (state[:,i] - goal)ᵀ Q (state[:,i] - goal)
     *         + control[:,i]ᵀ R control[:,i]
     *         + L_aug(state[:,i], control[:,i])
     * ```
     *
     * @param state 状态矩阵 (6 × N), 每列是一个状态
     * @param control 控制矩阵 (2 × N), 每列是一个控制
     * @return 代价向量 (N × 1), 每个元素是对应状态-控制对的代价
     *
     * ## 向量化计算流程
     *
     * 1. 计算状态误差矩阵: error = state - goal (广播)
     * 2. 计算加权误差: weighted_error = Q .* error (逐元素乘法)
     * 3. 计算状态代价: cost_q[i] = Σⱼ error[j,i] * weighted_error[j,i]
     * 4. 类似地计算控制代价: cost_r[i] = Σⱼ control[j,i] * (R .* control)[j,i]
     * 5. 添加约束惩罚代价
     *
     * @note override 关键字表示覆盖基类的虚函数
     * @note 使用 Eigen 的数组运算和列求和来实现高效的并行计算
     * @note 在并行线搜索中,此函数会被多次调用
     *
     * @see parallel_dynamics() 批量动力学计算
     * @see cost() 单个状态-控制对的代价计算
     *
     * @example
     * ```cpp
     * // 同时评估 8 个候选轨迹点的代价
     * Eigen::Matrix<double, 6, 8> states;
     * Eigen::Matrix<double, 2, 8> controls;
     * // ... 初始化 ...
     *
     * Eigen::Matrix<double, 8, 1> costs = node.parallel_cost(states, controls);
     * // costs[i] 是第 i 个状态-控制对的代价
     * ```
     */
    Eigen::Matrix<double, PARALLEL_NUM, 1> parallel_cost(const Eigen::Ref<const Eigen::Matrix<double, 6, PARALLEL_NUM>>& state,
                                                         const Eigen::Ref<const Eigen::Matrix<double, 2, PARALLEL_NUM>>& control) override {
        // 计算状态误差 (广播目标状态到 N 列)
        Eigen::Matrix<double, 6, PARALLEL_NUM> error = state - this->goal_.replicate(1, PARALLEL_NUM);

        // 准备权重矩阵 (将对角元素复制为 N 列)
        Eigen::Array<double, 6, PARALLEL_NUM> Q_array = (Q_.diagonal().replicate(1, PARALLEL_NUM)).array();
        Eigen::Array<double, 6, PARALLEL_NUM> error_array = error.array();
        Eigen::Array<double, 2, PARALLEL_NUM> R_array = (R_.diagonal().replicate(1, PARALLEL_NUM)).array();
        Eigen::Array<double, 2, PARALLEL_NUM> control_array = control.array();

        // 计算加权误差: Qe 和 Ru (逐元素乘法)
        Eigen::Array<double, 6, PARALLEL_NUM> new_error_array = (error_array * Q_array);
        Eigen::Array<double, 2, PARALLEL_NUM> new_control_array = (R_array * control_array);

        // 计算状态代价: eᵀQe (对每列求和)
        Eigen::Matrix<double, PARALLEL_NUM, 1> cost_q = (error_array * new_error_array).matrix().colwise().sum().transpose();
        // 计算控制代价: uᵀRu (对每列求和)
        Eigen::Matrix<double, PARALLEL_NUM, 1> cost_r = (new_control_array * control_array).matrix().colwise().sum().transpose();

        // 合并状态代价和控制代价
        Eigen::Matrix<double, PARALLEL_NUM, 1> ans1 = cost_q + cost_r;

        // 添加约束惩罚代价 (批量计算)
        Eigen::Matrix<double, PARALLEL_NUM, 1> ans2 = constraints_.parallel_augmented_lagrangian_cost(state, control);

        return ans1 + ans2;
    }

    /**
     * @brief 计算代价函数的雅可比矩阵 (一阶导数)
     *
     * 计算代价函数对状态和控制的偏导数,用于 iLQR 的 Backward Pass。
     *
     * 雅可比矩阵:
     * ```
     * ∂cost/∂x = 2Q(x - x_goal) + ∂L_aug/∂x
     * ∂cost/∂u = 2Ru + ∂L_aug/∂u
     * ```
     *
     * 其中:
     * - `2Q(x - x_goal)`: 状态跟踪代价的梯度
     * - `2Ru`: 控制代价的梯度
     * - `∂L_aug/∂x, ∂L_aug/∂u`: 增广拉格朗日项的梯度 (约束惩罚)
     *
     * ## 梯度的物理意义
     *
     * - `∂cost/∂x`: 指示在当前状态下,沿哪个方向移动可以最快降低代价
     * - `∂cost/∂u`: 指示如何调整控制输入可以最快降低代价
     *
     * ## 在 iLQR 中的应用
     *
     * 梯度用于计算 Q 函数的一阶项:
     * ```
     * Qₓ = lₓ + fₓᵀ Vₓ'
     * Qᵤ = lᵤ + fᵤᵀ Vₓ'
     * ```
     *
     * @param state 当前状态 (6×1)
     * @param control 当前控制 (2×1)
     * @return std::pair<Jx, Ju>
     *         - Jx: ∂cost/∂x (6×1)
     *         - Ju: ∂cost/∂u (2×1)
     *
     * @note 对于二次代价 xᵀQx, 梯度为 2Qx
     * @note 约束项的梯度由约束对象自动计算
     *
     * @example
     * ```cpp
     * auto [Jx, Ju] = node.cost_jacobian(x, u);
     * // Jx[0] 是代价对 x 位置的偏导数
     * // Ju[0] 是代价对转向角速度的偏导数
     * ```
     */
    std::pair<Eigen::Matrix<double, 6, 1>, Eigen::Matrix<double, 2, 1>>
    cost_jacobian(const Eigen::Ref<const VectorState>& state,
                  const Eigen::Ref<const VectorControl>& control) override {
        VectorState state_error = state - this->goal_;
        // 状态跟踪代价的梯度: ∂[(x-g)ᵀQ(x-g)]/∂x = 2Q(x-g)
        Eigen::Matrix<double, 6, 1> Jx = 2 * Q_ * state_error;
        // 控制代价的梯度: ∂[uᵀRu]/∂u = 2Ru
        Eigen::Matrix<double, 2, 1> Ju = 2 * R_ * control;
        // 添加约束惩罚项的梯度
        auto constraints_jacobian = constraints_.augmented_lagrangian_jacobian(state, control);
        Jx += constraints_jacobian.first;
        Ju += constraints_jacobian.second;

        return {Jx, Ju};
    }

    /**
     * @brief 计算代价函数的 Hessian 矩阵 (二阶导数)
     *
     * 计算代价函数对状态和控制的二阶偏导数,用于 iLQR 的 Backward Pass。
     *
     * Hessian 矩阵:
     * ```
     * ∂²cost/∂x² = 2Q + ∂²L_aug/∂x²
     * ∂²cost/∂u² = 2R + ∂²L_aug/∂u²
     * ```
     *
     * 其中:
     * - `2Q`: 状态跟踪代价的 Hessian (常数矩阵)
     * - `2R`: 控制代价的 Hessian (常数矩阵)
     * - `∂²L_aug/∂x², ∂²L_aug/∂u²`: 增广拉格朗日项的 Hessian (与约束相关)
     *
     * ## 在 iLQR 中的作用
     *
     * Hessian 用于计算 Q 函数的二阶项:
     * ```
     * Qₓₓ = lₓₓ + fₓᵀ Vₓₓ fₓ + ...
     * Qᵤᵤ = lᵤᵤ + fᵤᵀ Vₓₓ fᵤ
     * ```
     *
     * 其中 lₓₓ 和 lᵤᵤ 就是此函数返回的 Hessian 矩阵。
     *
     * @param state 当前状态 (6×1)
     * @param control 当前控制 (2×1)
     * @return std::pair<Hx, Hu>
     *         - Hx: ∂²cost/∂x² (6×6), 状态 Hessian 矩阵
     *         - Hu: ∂²cost/∂u² (2×2), 控制 Hessian 矩阵
     *
     * @note 对于二次代价 xᵀQx, Hessian 为常数矩阵 2Q
     * @note 约束项的 Hessian 由约束对象自动计算
     * @note Hessian 矩阵是对称正定的 (确保 Q 函数是凸的)
     *
     * @see cost_jacobian() 代价函数的梯度
     * @see Backward() 使用 Hessian 的反向传播算法
     */
    std::pair<MatrixQ, MatrixR> cost_hessian(const Eigen::Ref<const VectorState>& state,
                                             const Eigen::Ref<const VectorControl>& control) override {
        // 状态代价的 Hessian: ∂²[(x-g)ᵀQ(x-g)]/∂x² = 2Q
        MatrixQ Hx = 2 * Q_;
        // 控制代价的 Hessian: ∂²[uᵀRu]/∂u² = 2R
        MatrixR Hu = 2 * R_;

        // 添加约束惩罚项的 Hessian
        auto constraints_hessian = constraints_.augmented_lagrangian_hessian(state, control);
        Hx += std::get<0>(constraints_hessian);  // 添加 ∂²L_aug/∂x²
        Hu += std::get<1>(constraints_hessian);  // 添加 ∂²L_aug/∂u²

        return {Hx, Hu};
    }

    /**
     * @brief 更新拉格朗日乘子 λ
     *
     * 根据当前状态和控制的约束违反量,更新拉格朗日乘子。
     * 这是增广拉格朗日法的核心步骤之一。
     *
     * 更新公式:
     * ```
     * λ_new = λ_old + μ * c(x, u)
     * ```
     *
     * 其中:
     * - `λ`: 拉格朗日乘子向量 (约束维度)
     * - `μ`: 惩罚系数 (标量)
     * - `c(x, u)`: 约束违反量 (正数表示违反约束)
     *
     * ## 物理意义
     *
     * - λ 可以理解为约束的"影子价格"或"拉格朗日压力"
     * - 当约束被违反时,λ 增加,使得代价函数更加惩罚约束违反
     * - 当约束满足时,λ 保持不变或减小
     *
     * @param state 当前状态 (6×1)
     * @param control 当前控制 (2×1)
     *
     * @note 此函数在外层增广拉格朗日迭代的每次循环中调用
     * @note 实际更新由约束对象的 update_lambda 方法执行
     *
     * @see UpdateLambda() 调用此函数的求解器方法
     * @see update_mu() 更新惩罚系数
     */
    void update_lambda(const Eigen::Ref<const VectorState>& state,
                               const Eigen::Ref<const VectorControl>& control) override {
        constraints_.update_lambda(state, control);
    }

    /**
     * @brief 更新惩罚系数 μ
     *
     * 设置新的惩罚系数,用于增广拉格朗日法。
     *
     * @param new_mu 新的惩罚系数 (通常是旧值的倍数,如 μ_new = 100 * μ_old)
     *
     * ## 使用场景
     *
     * 当约束违反严重时 (violation > 5 * tolerance),需要大幅增加 μ:
     * - 这会增强约束惩罚项的权重
     * - 迫使优化器更快地满足约束
     *
     * @note μ 不能过大,否则会导致数值不稳定
     * @note μ 的典型值范围: 1.0 ~ 10000.0
     *
     * @see UpdateMu() 调用此函数的求解器方法
     * @see update_lambda() 更新拉格朗日乘子
     */
    void update_mu(double new_mu) {
        constraints_.update_mu(new_mu);
    }

    /**
     * @brief 重置拉格朗日乘子 λ 为零
     *
     * 在开始新的优化问题前,将所有拉格朗日乘子重置为 0。
     *
     * @note 在 linearizedInitialGuess() 中调用
     * @note λ = 0 表示没有约束的"记忆"
     */
    void reset_lambda() {
        auto dim_c = constraints_.get_constraint_dim();  // 获取约束维度
        Eigen::Matrix<double, Eigen::Dynamic, 1> result(dim_c);
        result.setZero();  // 设置为零向量
        constraints_.set_lambda(result);
    }

    /**
     * @brief 重置惩罚系数 μ 为初始值
     *
     * 将惩罚系数重置为默认值 1.0,用于开始新的优化问题。
     *
     * @note 在 linearizedInitialGuess() 中调用
     * @note μ = 1.0 是一个合理的初始值
     */
    void reset_mu() {
        constraints_.set_mu(1.0);
    }

    /**
     * @brief 计算最大约束违反量
     *
     * 计算当前状态-控制对的所有约束违反量中的最大值。
     *
     * 约束违反量定义:
     * ```
     * violation = max(0, c(x, u))
     * ```
     *
     * 其中 c(x, u) 是约束函数,c > 0 表示违反约束。
     *
     * @param state 当前状态 (6×1)
     * @param control 当前控制 (2×1)
     * @return 最大约束违反量 (标量)
     *         - 0: 所有约束都满足
     *         - > 0: 至少有一个约束被违反
     *
     * ## 应用
     *
     * 用于判断增广拉格朗日法是否收敛:
     * ```cpp
     * if (max_violation < tolerance) {
     *     // 收敛,退出外层循环
     * }
     * ```
     *
     * @note 此函数在每次外层迭代中调用
     * @note 返回值越小,表示约束满足得越好
     *
     * @see ComputeConstraintViolation() 调用此函数的求解器方法
     */
    double max_constraints_violation(const Eigen::Ref<const VectorState>& state,
                               const Eigen::Ref<const VectorControl>& control) const override {
        return constraints_.max_violation(state, control);
    }

    /**
     * @brief 动态更新约束 (添加新的线性约束)
     *
     * 在优化过程中动态添加新的线性约束,主要用于障碍物避障。
     *
     * 约束形式:
     * ```
     * A * x <= C
     * ```
     *
     * 其中:
     * - A: 1×6 系数矩阵 (行向量)
     * - x: 6×1 状态向量
     * - C: 标量常数
     *
     * ## 应用场景
     *
     * 当车辆进入障碍物区域时,动态添加约束:
     * - 左侧障碍物: -y <= y_max  即  y <= y_max
     * - 右侧障碍物: y <= -y_min  即  y >= y_min
     *
     * @param A_rows 约束系数向量 (1×6)
     * @param C_rows 约束常数 (标量)
     *
     * @note 此函数由 UpdateConstraints() 在每次 iLQR 迭代中调用
     * @note 约束会累积添加,直到求解器重置
     *
     * @see UpdateConstraints() 障碍物约束更新逻辑
     *
     * @example
     * ```cpp
     * // 添加约束 y <= 5.0 (不能超过左边界)
     * Eigen::Matrix<double, 1, 6> A;
     * A << 0, -1, 0, 0, 0, 0;  // -y
     * double C = 5.0;
     * node.update_constraints(A, C);
     * ```
     */
    void update_constraints(const Eigen::Ref<const Eigen::Matrix<double, 1, 6>> A_rows, double C_rows) override {
        constraints_.UpdateConstraints(A_rows, C_rows);
    }

public:
    // ============================================
    // 公共成员变量
    // ============================================

    ConstraintsType constraints_;  ///< 约束对象 (盒式约束、二次约束或线性约束)


protected:
    // ============================================
    // 受保护成员变量 (模型参数)
    // ============================================

    double L_;   ///< 车辆轴距 (m), 前后轮轴之间的距离, 决定了车辆的转向特性
    double dt_;  ///< 时间步长 (s), 离散化时间间隔, 影响积分精度和实时性
    double k_;   ///< 速度修正系数 (无量纲), 用于高速动力学修正: θ̇ = v*tan(δ)/[L*(1+k*v²)]

    MatrixQ Q_;  ///< 状态代价权重矩阵 (6×6 对角矩阵), 定义各状态维度的跟踪重要性
    MatrixR R_;  ///< 控制代价权重矩阵 (2×2 对角矩阵), 定义控制平滑性的重要性

    // 以下变量用于缓存增广拉格朗日项的导数信息 (可能未使用)
    Eigen::Matrix<double, 6, 1> aug_dx_;   ///< 增广拉格朗日代价对状态的梯度缓存
    Eigen::Matrix<double, 2, 1> aug_du_;   ///< 增广拉格朗日代价对控制的梯度缓存
    Eigen::Matrix<double, 6, 6> aug_dxx_;  ///< 增广拉格朗日代价对状态的 Hessian 缓存
    Eigen::Matrix<double, 2, 2> aug_duu_;  ///< 增广拉格朗日代价对控制的 Hessian 缓存
    Eigen::Matrix<double, 6, 2> aug_dxu_;  ///< 增广拉格朗日代价的交叉 Hessian 缓存
    double aug_cost_ = 0.0;                 ///< 增广拉格朗日代价值缓存

};

// ============================================
// 文件结束
// ============================================

#endif // NEW_BICYCLE_NODE_H_
