/**
 * @file new_al_ilqr.h
 * @brief 增广拉格朗日 iLQR (Augmented Lagrangian iterative Linear Quadratic Regulator) 求解器
 *
 * 本文件实现了基于增广拉格朗日方法的 iLQR 算法,用于求解带约束的最优控制问题。
 *
 * 算法采用双层迭代结构:
 * - 外层循环: 通过增广拉格朗日法处理约束,更新拉格朗日乘子 λ 和惩罚系数 μ
 * - 内层循环: 标准 iLQR 算法 (Backward-Forward 迭代)
 *
 * 主要特性:
 * - 支持动态线性约束 (如动态障碍物避障)
 * - 并行线搜索优化 (同时评估多个步长)
 * - RK2 数值积分
 * - 二阶导数信息 (Hessian) 用于更快收敛
 */

#ifndef NEW_ALILQR_H
#define NEW_ALILQR_H

#include <array>
#include <vector>
#include <Eigen/Dense>
#include <tuple>
#include <memory>
#include "model/new_bicycle_node.h"
#include "constraints/box_constraints.h"
#include "constraints/quadratic_constraints.h"
#include <iostream>
#include <chrono>


/**
 * @brief 增广拉格朗日 iLQR 求解器类
 *
 * @tparam state_dim 状态维度 (例如: 6 表示 [x, y, θ, δ, v, a])
 * @tparam control_dim 控制维度 (例如: 2 表示 [δ_rate, a_rate])
 *
 * 本类实现了完整的增广拉格朗日 iLQR 算法,用于求解以下优化问题:
 *
 * minimize   Σ L(x_t, u_t) + L_f(x_N)
 * subject to x_{t+1} = f(x_t, u_t)              (动力学约束)
 *            c(x_t, u_t) <= 0                    (不等式约束)
 *
 * 其中:
 * - L(x, u): 阶段代价函数
 * - L_f(x): 终端代价函数
 * - f(x, u): 系统动力学方程
 * - c(x, u): 约束函数
 */
template<int state_dim, int control_dim>
class NewALILQR {
public:
    // ============================================
    // 类型别名定义
    // ============================================
    using VectorState = Eigen::Matrix<double, state_dim, 1>;              ///< 状态向量类型
    using VectorControl = Eigen::Matrix<double, control_dim, 1>;          ///< 控制向量类型
    using VectorConstraint = Eigen::Matrix<double, 2 * (control_dim + state_dim), 1>; ///< 约束向量类型 (盒式约束)
    using MatrixA = Eigen::Matrix<double, state_dim, state_dim>;          ///< 动力学雅可比矩阵 ∂f/∂x
    using MatrixB = Eigen::Matrix<double, state_dim, control_dim>;        ///< 动力学雅可比矩阵 ∂f/∂u
    using MatrixQ = Eigen::Matrix<double, state_dim, state_dim>;          ///< 状态代价 Hessian 矩阵
    using MatrixR = Eigen::Matrix<double, control_dim, control_dim>;      ///< 控制代价 Hessian 矩阵
    using MatrixK = Eigen::Matrix<double, control_dim, state_dim>;        ///< 反馈增益矩阵 (u = K*x + k)

    // ============================================
    // 构造函数
    // ============================================

    /**
     * @brief 基础构造函数 (不包含障碍物约束)
     *
     * @param ilqr_nodes iLQR 节点列表,每个节点包含一个时间步的代价函数、动力学和约束
     *                   长度为 horizon + 1 (包括终端节点)
     * @param init_state 初始状态 x_0
     *
     * 本构造函数初始化所有必要的数据结构:
     * - 轨迹存储 (x_list_, u_list_)
     * - 导数信息 (雅可比和 Hessian 矩阵)
     * - 反馈增益 (K_list_, k_list_)
     * - 约束违反列表
     */
    NewALILQR(const std::vector<std::shared_ptr<NewILQRNode<state_dim, control_dim>>>& ilqr_nodes,
        const VectorState& init_state)
        : ilqr_nodes_(ilqr_nodes), init_state_(init_state) {
        zero_control_.setZero();      // 零控制向量 (用于终端节点)
        zero_state_.setZero();         // 零状态向量
        horizon_ = ilqr_nodes.size() - 1;  // 时域长度 = 节点数 - 1

        // 分配轨迹存储空间
        x_list_.resize(state_dim, horizon_ + 1);    // 状态轨迹: 每列是一个时间步的状态
        u_list_.resize(control_dim, horizon_);      // 控制序列: 每列是一个时间步的控制

        // 分配导数信息存储空间
        cost_augmented_lagrangian_jacobian_x_list_.resize(horizon_ + 1);   // 增广拉格朗日代价关于状态的梯度
        cost_augmented_lagrangian_jacobian_u_list_.resize(horizon_);       // 增广拉格朗日代价关于控制的梯度
        cost_augmented_lagrangian_hessian_x_list_.resize(horizon_ + 1);    // 增广拉格朗日代价关于状态的 Hessian
        cost_augmented_lagrangian_hessian_u_list_.resize(horizon_);        // 增广拉格朗日代价关于控制的 Hessian
        dynamics_jacobian_x_list_.resize(horizon_);                        // 动力学雅可比 ∂f/∂x
        dynamics_jacobian_u_list_.resize(horizon_);                        // 动力学雅可比 ∂f/∂u

        // 分配约束和优化相关存储空间
        max_constraints_violation_list_.resize(horizon_ + 1, 1);  // 每个时间步的最大约束违反量
        K_list_.resize(horizon_);                                  // 反馈增益矩阵列表
        k_list_.resize(horizon_);                                  // 前馈控制列表
        dynamics_hession_x_list_.resize(horizon_);                 // 动力学 Hessian 张量
        cost_list_.resize(horizon_ + 1);                           // 每个时间步的代价

        // 障碍物约束标志初始化
        obs_constraints_ = false;
        left_obs_size_ = 0;
        right_obs_size_ = 0;
    }

    /**
     * @brief 带障碍物约束的构造函数
     *
     * @param ilqr_nodes iLQR 节点列表
     * @param init_state 初始状态
     * @param left_obs 左侧障碍物列表,每个障碍物用 2×4 矩阵表示 (4个顶点的 [x, y] 坐标)
     * @param right_obs 右侧障碍物列表,每个障碍物用 2×4 矩阵表示
     *
     * 障碍物表示为矩形,通过4个顶点定义:
     * - 每列是一个顶点: [x; y]
     * - 顶点按逆时针或顺时针顺序排列
     *
     * 避障策略:
     * - 左侧障碍物: 添加约束 y <= y_max (车辆不能超过左边界)
     * - 右侧障碍物: 添加约束 y >= y_min (车辆不能超过右边界)
     *
     * 约束动态激活: 只有当车辆进入障碍物的包围盒时,才会激活对应的约束
     * (通过 UpdateConstraints() 函数判断点在矩形内部)
     */
    NewALILQR(const std::vector<std::shared_ptr<NewILQRNode<state_dim, control_dim>>>& ilqr_nodes,
        const VectorState& init_state,
        const std::vector<Eigen::Matrix<double, 2, 4>>& left_obs,
        const std::vector<Eigen::Matrix<double, 2, 4>>& right_obs)
        : NewALILQR(ilqr_nodes, init_state) {  // 委托构造,先调用基础构造函数

        // 预计算每个障碍物的边界值
        l_obs_y_max_.clear();
        r_obs_y_min_.clear();
        for (auto element : left_obs) {
            l_obs_y_max_.push_back(element.row(1).maxCoeff());  // 左侧障碍物的最大 y 值
        }
        for (auto element : right_obs) {
            r_obs_y_min_.push_back(element.row(1).minCoeff());  // 右侧障碍物的最小 y 值
        }

        // 设置障碍物约束标志
        obs_constraints_ = true;
        left_obs_size_ = left_obs.size();
        right_obs_size_ = right_obs.size();
        obs_constraints_ = ((left_obs_size_ != 0) || (right_obs_size_ != 0));

        // 预处理左侧障碍物的几何信息
        // 使用矢量叉积方法判断点是否在矩形内部 (需要预计算边向量)
        if (left_obs_size_ > 0) {
            // 为所有左侧障碍物的顶点分配存储空间 (每列对应一个障碍物)
            l_point1_ = Eigen::MatrixXd(2, left_obs_size_);
            l_point2_ = l_point1_;
            l_point3_ = l_point1_;
            l_point4_ = l_point1_;
            l_vector1_ = l_point1_;  // 边向量: point2 - point1
            l_vector2_ = l_point2_;  // 边向量: point3 - point2
            l_vector3_ = l_point3_;  // 边向量: point4 - point3
            l_vector4_ = l_point4_;  // 边向量: point1 - point4

            // 提取所有障碍物的顶点坐标
            for (int index = 0; index < left_obs_size_; ++index) {
                l_point1_.col(index) = left_obs[index].col(0);  // 第1个顶点
                l_point2_.col(index) = left_obs[index].col(1);  // 第2个顶点
                l_point3_.col(index) = left_obs[index].col(2);  // 第3个顶点
                l_point4_.col(index) = left_obs[index].col(3);  // 第4个顶点
            }

            // 计算矩形的4条边向量 (用于后续的点在多边形内测试)
            l_vector1_ = l_point2_ - l_point1_;
            l_vector2_ = l_point3_ - l_point2_;
            l_vector3_ = l_point4_ - l_point3_;
            l_vector4_ = l_point1_ - l_point4_;
        }


        // 预处理右侧障碍物的几何信息 (逻辑与左侧相同)
        if (right_obs_size_ > 0) {
            r_point1_ = Eigen::MatrixXd(2, right_obs_size_);
            r_point2_ = r_point1_;
            r_point3_ = r_point1_;
            r_point4_ = r_point1_;
            r_vector1_ = r_point1_;
            r_vector2_ = r_point2_;
            r_vector3_ = r_point3_;
            r_vector4_ = r_point4_;

            for (int index = 0; index < right_obs_size_; ++index) {
                r_point1_.col(index) = right_obs[index].col(0);
                r_point2_.col(index) = right_obs[index].col(1);
                r_point3_.col(index) = right_obs[index].col(2);
                r_point4_.col(index) = right_obs[index].col(3);
            }

            r_vector1_ = r_point2_ - r_point1_;
            r_vector2_ = r_point3_ - r_point2_;
            r_vector3_ = r_point4_ - r_point3_;
            r_vector4_ = r_point1_ - r_point4_;
        }
    }

    // ============================================
    // 辅助函数
    // ============================================

    /**
     * @brief 批量计算2D向量叉积 (z分量)
     *
     * @param v1_series 第一组向量,2×N 矩阵,每列是一个2D向量
     * @param v2_series 第二组向量,2×N 矩阵,每列是一个2D向量
     * @return N维数组,包含每对向量叉积的 z 分量: v1_x * v2_y - v1_y * v2_x
     *
     * 用途: 判断点是否在多边形内部
     * - 如果点 P 在边 AB 的左侧,则 (P-A) × (B-A) > 0
     * - 如果点在凸多边形内部,则点在所有边的同一侧 (所有叉积同号)
     */
    Eigen::ArrayXd MultiVectorCross(const Eigen::MatrixXd& v1_series, const Eigen::MatrixXd& v2_series) {
        Eigen::ArrayXd v1_x = v1_series.row(0).transpose().array();  // 提取 x 分量
        Eigen::ArrayXd v1_y = v1_series.row(1).transpose().array();  // 提取 y 分量
        Eigen::ArrayXd v2_x = v2_series.row(0).transpose().array();
        Eigen::ArrayXd v2_y = v2_series.row(1).transpose().array();
        Eigen::ArrayXd ans = v1_x * v2_y - v1_y * v2_x;  // 叉积的 z 分量
        return ans;
    }


    // ============================================
    // 核心算法函数声明
    // ============================================

    /**
     * @brief 使用线性化 LQR 生成初始轨迹猜测
     *
     * 通过离散时间 Riccati 方程反向传播计算初始反馈增益,然后前向传播生成初始轨迹。
     * 这比随机初始化能提供更好的起点,加快收敛速度。
     */
    void linearizedInitialGuess();

    /**
     * @brief 更新动态约束 (障碍物避障约束)
     *
     * 根据当前轨迹 x_list_,判断每个时间步是否进入障碍物区域。
     * 如果进入,则向对应节点添加线性约束 (y <= y_max 或 y >= y_min)。
     */
    void UpdateConstraints();

    /**
     * @brief 计算指定范围内的导数信息
     * @param start 起始时间步索引
     * @param end 结束时间步索引
     */
    void CalcDerivatives(int start, int end);

    /**
     * @brief 计算所有时间步的导数信息
     *
     * 包括:
     * - 增广拉格朗日代价函数的梯度和 Hessian
     * - 动力学的雅可比矩阵
     * - 动力学的 Hessian 张量 (用于更精确的二阶近似)
     */
    void CalcDerivatives();

    /** @brief 计算总代价 (所有时间步代价之和) */
    double computeTotalCost();

    /**
     * @brief 使用给定步长更新轨迹和代价列表
     * @param alpha 线搜索步长 (0 < alpha <= 1)
     */
    void UpdateTrajectoryAndCostList(double alpha);

    /**
     * @brief 并行线搜索:同时评估多个步长,选择最优的
     *
     * @param alpha 初始步长
     * @param best_alpha [输出] 最优步长
     * @param best_cost [输出] 最优步长对应的代价
     *
     * 性能优化: 通过 PARALLEL_NUM 宏控制并行度,默认同时评估多个步长
     * (例如: alpha, alpha/3, alpha/9, ...)
     */
    void ParallelLinearSearch(double alpha, double& best_alpha, double& best_cost);

    /**
     * @brief 批量计算系统动力学 (用于并行线搜索)
     *
     * @param x 状态矩阵,state_dim × PARALLEL_NUM (每列是一个状态)
     * @param u 控制矩阵,control_dim × PARALLEL_NUM (每列是一个控制)
     * @return 状态导数矩阵,state_dim × PARALLEL_NUM
     */
    Eigen::Matrix<double, state_dim, PARALLEL_NUM> State_Dot(const Eigen::Matrix<double, state_dim, PARALLEL_NUM>& x,
                                                             const Eigen::Matrix<double, control_dim, PARALLEL_NUM>& u);


    /**
     * @brief iLQR 向后传播 (Backward Pass)
     *
     * 从终端时刻 T 向后传播到初始时刻 0,计算每个时间步的:
     * - 反馈增益矩阵 K_t: 控制相对于状态的线性反馈
     * - 前馈控制 k_t: 开环控制修正
     *
     * 基于动态规划的 Bellman 方程:
     * Q_u = l_u + B^T V_x
     * Q_uu = l_uu + B^T V_xx B
     * K = -Q_uu^{-1} Q_ux
     * k = -Q_uu^{-1} Q_u
     *
     * 其中 V 是值函数,l 是阶段代价,B 是控制雅可比矩阵
     */
    void Backward();

    /**
     * @brief iLQR 向前传播 (Forward Pass) 带线搜索
     *
     * 使用 Backward Pass 计算的 K 和 k,通过线搜索更新轨迹:
     * u_new = u_old + K*(x_new - x_old) + alpha*k
     *
     * 如果简单线搜索失败,则调用并行线搜索 ParallelLinearSearch()
     */
    void Forward();

    /**
     * @brief 更新增广拉格朗日法的惩罚系数 μ
     * @param gain 增益因子 (新的 μ = 旧的 μ * gain)
     *
     * 当约束违反严重时,增加 μ 来加强约束的惩罚力度
     */
    void UpdateMu(double gain);

    /**
     * @brief 更新拉格朗日乘子 λ
     *
     * 对每个时间步,根据当前约束违反量更新拉格朗日乘子:
     * λ_new = λ_old + μ * c(x, u)
     *
     * 其中 c(x, u) 是约束函数,μ 是惩罚系数
     */
    void UpdateLambda();

    /**
     * @brief 计算整条轨迹的最大约束违反量
     * @return max_t { max_i { c_i(x_t, u_t) } }
     *
     * 用于判断是否满足约束容差,决定外层迭代是否收敛
     */
    double ComputeConstraintViolation();

    /**
     * @brief 主优化函数:增广拉格朗日 iLQR 的双层迭代
     *
     * @param max_outer_iter 外层最大迭代次数 (增广拉格朗日迭代)
     * @param max_inner_iter 内层最大迭代次数 (iLQR 迭代)
     * @param max_violation 约束违反容差 (当所有约束违反 < max_violation 时停止)
     *
     * 算法流程:
     * 1. 初始化: linearizedInitialGuess()
     * 2. 外层循环 (最多 max_outer_iter 次):
     *    a. 调用 ILQRProcess() 执行 iLQR 内层迭代
     *    b. 计算约束违反量
     *    c. 如果违反量 < max_violation,收敛退出
     *    d. 否则,更新 λ 或 μ,继续外层迭代
     */
    void optimize(int max_outer_iter, int max_inner_iter, double max_violation);

    /**
     * @brief iLQR 内层迭代过程
     *
     * @param max_iter 最大迭代次数
     * @param max_tol 代价改善容差 (当代价改善 < max_tol 时停止)
     *
     * 每次迭代执行:
     * 1. UpdateConstraints(): 更新动态约束
     * 2. CalcDerivatives(): 计算导数
     * 3. Backward(): 计算反馈增益
     * 4. Forward(): 线搜索更新轨迹
     */
    void ILQRProcess(int max_iter, double max_tol);

    // ============================================
    // 访问函数
    // ============================================

    /** @brief 获取优化后的状态轨迹 (state_dim × (horizon+1)) */
    Eigen::MatrixXd get_x_list() { return x_list_; }

    /** @brief 获取优化后的控制序列 (control_dim × horizon) */
    Eigen::MatrixXd get_u_list() { return u_list_; }

    /** @brief 获取反馈增益矩阵列表 */
    std::vector<MatrixK> get_K() {return K_list_; }

    /** @brief 获取前馈控制列表 */
    std::vector<VectorControl> get_k() {return k_list_; }

    /** @brief 获取动力学雅可比 ∂f/∂x 列表 */
    std::vector<Eigen::Matrix<double, state_dim, state_dim>> get_jacobian_x() { return dynamics_jacobian_x_list_; }

    /** @brief 获取动力学雅可比 ∂f/∂u 列表 */
    std::vector<Eigen::Matrix<double, state_dim, control_dim>> get_jacobian_u() { return dynamics_jacobian_u_list_; }




private:
    // ============================================
    // 私有成员变量
    // ============================================

    // 轨迹数据
    Eigen::MatrixXd x_list_;      ///< 当前状态轨迹 (state_dim × (horizon+1))
    Eigen::MatrixXd u_list_;      ///< 当前控制序列 (control_dim × horizon)
    Eigen::MatrixXd pre_x_list_;  ///< 上一次迭代的状态轨迹 (用于线搜索回退)
    Eigen::MatrixXd pre_u_list_;  ///< 上一次迭代的控制序列 (用于线搜索回退)

    // 常用零向量 (避免重复分配)
    Eigen::Matrix<double, control_dim, 1> zero_control_;  ///< 零控制向量
    Eigen::Matrix<double, state_dim, 1> zero_state_;      ///< 零状态向量

    // 增广拉格朗日代价函数的导数信息
    std::vector<Eigen::Matrix<double, state_dim, 1>> cost_augmented_lagrangian_jacobian_x_list_;     ///< ∂L_aug/∂x
    std::vector<Eigen::Matrix<double, control_dim, 1>> cost_augmented_lagrangian_jacobian_u_list_;   ///< ∂L_aug/∂u
    std::vector<Eigen::Matrix<double, state_dim, state_dim>> cost_augmented_lagrangian_hessian_x_list_;   ///< ∂²L_aug/∂x²
    std::vector<Eigen::Matrix<double, control_dim, control_dim>> cost_augmented_lagrangian_hessian_u_list_; ///< ∂²L_aug/∂u²

    // 动力学的导数信息
    std::vector<Eigen::Matrix<double, state_dim, state_dim>> dynamics_jacobian_x_list_;   ///< ∂f/∂x (状态转移矩阵 A)
    std::vector<Eigen::Matrix<double, state_dim, control_dim>> dynamics_jacobian_u_list_; ///< ∂f/∂u (控制输入矩阵 B)

    // 约束相关
    Eigen::MatrixXd max_constraints_violation_list_;  ///< 每个时间步的最大约束违反量

    // iLQR 反馈增益
    std::vector<MatrixK> K_list_;            ///< 反馈增益矩阵列表 (u = K*δx + k)
    std::vector<VectorControl> k_list_;      ///< 前馈控制列表

    // 值函数近似 (用于判断是否接受新轨迹)
    double deltaV_linear_ = 0.F;       ///< 线性预测的代价改善: k^T * Q_u
    double deltaV_quadratic_ = 0.F;    ///< 二次预测的代价改善: 0.5 * k^T * Q_uu * k

    // 增广拉格朗日参数
    double mu_ = 1.0;  ///< 惩罚系数 (初始值为 1.0)

    // 代价信息
    Eigen::VectorXd cost_list_;  ///< 每个时间步的代价

    // 动力学 Hessian 张量 (用于更精确的二阶近似)
    std::vector<std::tuple<MatrixA, MatrixA, MatrixA>> dynamics_hession_x_list_;  ///< ∂²f/∂x² (3个分量对应前3个状态维度)

public:
    // ============================================
    // 公共成员变量
    // ============================================

    // 核心数据
    std::vector<std::shared_ptr<NewILQRNode<state_dim, control_dim>>> ilqr_nodes_;  ///< iLQR 节点列表
    VectorState init_state_;  ///< 初始状态 x_0

    // 左侧障碍物几何信息 (2 × num_left_obs 矩阵,每列对应一个障碍物)
    Eigen::MatrixXd l_point1_;   ///< 左侧障碍物的第1个顶点 [x; y]
    Eigen::MatrixXd l_point2_;   ///< 左侧障碍物的第2个顶点
    Eigen::MatrixXd l_point3_;   ///< 左侧障碍物的第3个顶点
    Eigen::MatrixXd l_point4_;   ///< 左侧障碍物的第4个顶点

    Eigen::MatrixXd l_vector1_;  ///< 左侧障碍物的边向量: point2 - point1
    Eigen::MatrixXd l_vector2_;  ///< 左侧障碍物的边向量: point3 - point2
    Eigen::MatrixXd l_vector3_;  ///< 左侧障碍物的边向量: point4 - point3
    Eigen::MatrixXd l_vector4_;  ///< 左侧障碍物的边向量: point1 - point4

    // 右侧障碍物几何信息
    Eigen::MatrixXd r_point1_;   ///< 右侧障碍物的第1个顶点
    Eigen::MatrixXd r_point2_;   ///< 右侧障碍物的第2个顶点
    Eigen::MatrixXd r_point3_;   ///< 右侧障碍物的第3个顶点
    Eigen::MatrixXd r_point4_;   ///< 右侧障碍物的第4个顶点

    Eigen::MatrixXd r_vector1_;  ///< 右侧障碍物的边向量: point2 - point1
    Eigen::MatrixXd r_vector2_;  ///< 右侧障碍物的边向量: point3 - point2
    Eigen::MatrixXd r_vector3_;  ///< 右侧障碍物的边向量: point4 - point3
    Eigen::MatrixXd r_vector4_;  ///< 右侧障碍物的边向量: point1 - point4

    // 障碍物边界信息
    std::vector<double> l_obs_y_max_;  ///< 每个左侧障碍物的最大 y 值 (上边界)
    std::vector<double> r_obs_y_min_;  ///< 每个右侧障碍物的最小 y 值 (下边界)

    // 障碍物数量
    int left_obs_size_;   ///< 左侧障碍物数量
    int right_obs_size_;  ///< 右侧障碍物数量

    // 障碍物参数 (未使用,可能用于圆形障碍物)
    Eigen::VectorXd obs_y_;  ///< 障碍物中心 y 坐标
    Eigen::VectorXd obs_r_;  ///< 障碍物半径

    // 优化参数
    int horizon_ = 10;                ///< 时域长度 (默认10)
    bool obs_constraints_ = false;    ///< 是否存在障碍物约束

};


// ============================================
// 函数实现
// ============================================

/**
 * @brief UpdateConstraints 函数实现
 *
 * 动态约束更新算法:
 * 1. 遍历轨迹上的每个点 (x, y)
 * 2. 判断该点是否在障碍物包围盒内 (使用叉积法)
 * 3. 如果在盒内,添加相应的线性约束到该时间步的节点
 *
 * 点在凸多边形内的判断方法 (叉积法):
 * - 对于矩形的4条边,计算 (点-顶点) × (边向量)
 * - 如果所有叉积同号(这里都 < 0),则点在矩形内部
 */
template<int state_dim, int control_dim>
void NewALILQR<state_dim, control_dim>::UpdateConstraints() {

    // 提取轨迹的 x 和 y 坐标 (1 × (horizon+1) 向量)
    Eigen::MatrixXd xs = x_list_.row(0);  // x 坐标序列
    Eigen::MatrixXd ys = x_list_.row(1);  // y 坐标序列

    // 处理左侧障碍物约束
    if (left_obs_size_ > 0) {
        // 遍历轨迹上的每个时间步
        for (int index = 0; index < horizon_ + 1; ++index) {
            // 当前时间步的位置点 [x; y]
            Eigen::Matrix<double, 2, 1> points;
            points << xs(0,index), ys(0, index);

            // 将点复制 left_obs_size_ 次,以便批量处理所有障碍物
            Eigen::MatrixXd points_series = points.replicate(1, left_obs_size_);

            // 计算点到各障碍物顶点的向量 (2 × left_obs_size_ 矩阵)
            Eigen::MatrixXd p1 = points_series - l_point1_;  // 点 - 顶点1
            Eigen::MatrixXd p2 = points_series - l_point2_;  // 点 - 顶点2
            Eigen::MatrixXd p3 = points_series - l_point3_;  // 点 - 顶点3
            Eigen::MatrixXd p4 = points_series - l_point4_;  // 点 - 顶点4

            // 计算叉积,判断点相对于各条边的位置
            Eigen::ArrayXd p1_cross_lv1 = MultiVectorCross(p1, l_vector1_);  // (P-V1) × (V2-V1)
            Eigen::ArrayXd p2_cross_lv2 = MultiVectorCross(p2, l_vector2_);  // (P-V2) × (V3-V2)
            Eigen::ArrayXd p3_cross_lv3 = MultiVectorCross(p3, l_vector3_);  // (P-V3) × (V4-V3)
            Eigen::ArrayXd p4_cross_lv4 = MultiVectorCross(p4, l_vector4_);  // (P-V4) × (V1-V4)

            // 点在矩形内的条件: 所有叉积 < 0 (点在所有边的同一侧)
            Eigen::Array<bool, Eigen::Dynamic, 1> ans = (p1_cross_lv1 < 0) && (p2_cross_lv2 < 0) && (p3_cross_lv3 < 0) && (p4_cross_lv4 < 0);

            // 找出点在哪些障碍物内部 (收集满足条件的索引)
            std::vector<int> true_indices;
            true_indices.clear();
            for (int i = 0; i < ans.size(); ++i) {
                if (ans[i]) {
                   true_indices.push_back(i);  // 第 i 个左侧障碍物包含当前点
                }
            }

            // 为每个包含当前点的障碍物添加约束
            Eigen::Matrix<double, 1, state_dim> A_rows;
            A_rows.setZero();
            A_rows(0, 1) = -1.0;  // 约束形式: -y <= y_max  即  y <= y_max (不能超过左边界)

            for (size_t i = 0; i < true_indices.size(); ++i) {
                int obs_index = true_indices[i];
                double y_max = l_obs_y_max_[obs_index];  // 该障碍物的上边界
                ilqr_nodes_[index]->update_constraints(A_rows, y_max);  // 动态添加约束到节点
            }
        }
    }

    // 处理右侧障碍物约束 (逻辑与左侧相同)
    if (right_obs_size_ > 0) {
        for (int index = 0; index < horizon_ + 1; ++index) {
            // 当前时间步的位置点
            Eigen::Matrix<double, 2, 1> points;
            points << xs(0,index), ys(0, index);

            // 批量处理所有右侧障碍物
            Eigen::MatrixXd points_series = points.replicate(1, right_obs_size_);

            // 计算点到各障碍物顶点的向量
            Eigen::MatrixXd p1 = points_series - r_point1_;
            Eigen::MatrixXd p2 = points_series - r_point2_;
            Eigen::MatrixXd p3 = points_series - r_point3_;
            Eigen::MatrixXd p4 = points_series - r_point4_;

            // 计算叉积判断点是否在矩形内
            Eigen::ArrayXd p1_cross_rv1 = MultiVectorCross(p1, r_vector1_);
            Eigen::ArrayXd p2_cross_rv2 = MultiVectorCross(p2, r_vector2_);
            Eigen::ArrayXd p3_cross_rv3 = MultiVectorCross(p3, r_vector3_);
            Eigen::ArrayXd p4_cross_rv4 = MultiVectorCross(p4, r_vector4_);

            // 判断点是否在右侧障碍物矩形内
            Eigen::Array<bool, Eigen::Dynamic, 1> ans = (p1_cross_rv1 < 0) && (p2_cross_rv2 < 0) && (p3_cross_rv3 < 0) && (p4_cross_rv4 < 0);

            // 收集包含该点的障碍物索引
            std::vector<int> true_indices;
            true_indices.clear();
            for (int i = 0; i < ans.size(); ++i) {
                if (ans[i]) {
                   true_indices.push_back(i);
                }
            }

            // 为每个包含当前点的右侧障碍物添加约束
            Eigen::Matrix<double, 1, state_dim> A_rows;
            A_rows.setZero();
            A_rows(0, 1) = 1.0;  // 约束形式: y <= -y_min  即  y >= y_min (不能超过右边界)

            for (size_t i = 0; i < true_indices.size(); ++i) {
                int obs_index = true_indices[i];
                double y_min = r_obs_y_min_[obs_index];  // 该障碍物的下边界
                ilqr_nodes_[index]->update_constraints(A_rows, -y_min);  // 动态添加约束
            }
        }
    }
}

/**
 * @brief linearizedInitialGuess 函数实现
 *
 * 使用离散时间 LQR (Linear Quadratic Regulator) 方法生成初始轨迹猜测。
 *
 * 算法步骤:
 * 1. Backward Pass: 通过 Riccati 方程反向传播计算反馈增益 K
 * 2. Forward Pass: 使用增益 K 前向仿真生成初始轨迹
 * 3. 重置增广拉格朗日参数 λ 和 μ
 *
 * 离散时间 Riccati 方程:
 * K_t = (R + B^T P_{t+1} B)^{-1} (B^T P_{t+1} A)
 * P_t = Q + A^T P_{t+1} (A - B K_t)
 *
 * 其中:
 * - Q: 状态代价 Hessian
 * - R: 控制代价 Hessian
 * - A, B: 在目标状态附近线性化的动力学雅可比
 * - P: 值函数的二次近似 (cost-to-go)
 */
template<int state_dim, int control_dim>
void NewALILQR<state_dim, control_dim>::linearizedInitialGuess() {
    // 初始化状态为给定的初始状态
    x_list_.col(0) = init_state_;

    // 初始化第一个控制为零 (可能未使用,因为会在后面覆盖)
    u_list_.col(0).setZero();

    // ========== Backward Pass: 计算 LQR 反馈增益 ==========
    // 终端代价的 Hessian 作为 Riccati 方程的边界条件 P_T = Q_T
    MatrixQ P = ilqr_nodes_[horizon_]->cost_hessian(zero_state_, zero_control_).first.Identity();

    // 从终端时刻向后传播到初始时刻
    for (int t = horizon_ - 1; t >= 0; --t) {
        // 在目标状态和零控制处线性化动力学: x_{t+1} ≈ A x_t + B u_t
        auto dynamics_jacobian = ilqr_nodes_[t]->dynamics_jacobian(ilqr_nodes_[t]->goal(), VectorControl::Zero());
        MatrixA A = dynamics_jacobian.first;   // ∂f/∂x
        MatrixB B = dynamics_jacobian.second;  // ∂f/∂u

        // 计算反馈增益: K = (R + B^T P B)^{-1} (B^T P A)
        // 注意: R 使用 Identity() * 20.0 是一个简化的正则化技巧
        MatrixK K = (ilqr_nodes_[t]->cost_hessian(zero_state_, zero_control_).second.Identity() * 20.0
                     + B.transpose() * P * B).inverse()
                    * (B.transpose() * P * A);
        K_list_[t] = K;

        // 更新 Riccati 方程: P_t = Q + A^T P_{t+1} (A - B K)
        P = ilqr_nodes_[t]->cost_hessian(zero_state_, zero_control_).first.Identity()
            + A.transpose() * P * (A - B * K);
    }

    // ========== Forward Pass: 使用 LQR 增益生成初始轨迹 ==========
    for (int t = 0; t < horizon_; ++t) {
        VectorState goal_state = ilqr_nodes_[t]->goal();  // 第 t 时刻的目标状态
        MatrixK K = K_list_[t];                            // 第 t 时刻的反馈增益

        // LQR 控制律: u = -K (x - x_goal)
        u_list_.col(t) = -K * (x_list_.col(t) - goal_state);

        // 仿真前向动力学: x_{t+1} = f(x_t, u_t)
        x_list_.col(t + 1) = ilqr_nodes_[t]->dynamics(x_list_.col(t), u_list_.col(t));
    }

    // ========== 重置增广拉格朗日参数 ==========
    // 开始新的优化前,需要将所有节点的拉格朗日乘子和惩罚系数重置
    for (auto node : ilqr_nodes_) {
        node->reset_lambda();  // λ = 0
        node->reset_mu();      // μ = 初始值 (通常为 1.0)
    }
}

/**
 * @brief CalcDerivatives 函数实现 (范围版本)
 *
 * 计算指定时间范围内所有节点的导数信息,为 iLQR 迭代做准备。
 *
 * @param start 起始时间步索引
 * @param end 结束时间步索引
 *
 * 对每个时间步 t ∈ [start, end],计算:
 * 1. 增广拉格朗日代价: L_aug(x, u) = L(x, u) + λ^T c(x, u) + (μ/2) ||c(x, u)||²
 * 2. 一阶导数 (梯度):
 *    - ∂L_aug/∂x: 关于状态的梯度
 *    - ∂L_aug/∂u: 关于控制的梯度
 * 3. 二阶导数 (Hessian):
 *    - ∂²L_aug/∂x²: 关于状态的 Hessian 矩阵
 *    - ∂²L_aug/∂u²: 关于控制的 Hessian 矩阵
 * 4. 动力学雅可比:
 *    - ∂f/∂x (A矩阵): 状态转移矩阵
 *    - ∂f/∂u (B矩阵): 控制输入矩阵
 * 5. 动力学 Hessian 张量: ∂²f/∂x² (用于更精确的二阶近似)
 */
template<int state_dim, int control_dim>
void NewALILQR<state_dim, control_dim>::CalcDerivatives(int start, int end) {
    // 用于性能分析的计时器 (已注释)
    // double cost_aug_hessian_time_sum = 0.0;
    // double cost_aug_jacobian_time_sum = 0.0;

    // 遍历指定范围内的所有时间步
    for (int index = start; index <= end; ++index) {
        auto x = x_list_.col(index);  // 第 index 时刻的状态
        auto u = u_list_.col(index);  // 第 index 时刻的控制

        // 计算当前时刻的增广拉格朗日代价
        cost_list_[index] = ilqr_nodes_[index]->cost(x, u);

        // 计算增广拉格朗日代价的一阶导数 (梯度)
        //auto start_cost_jacobian = std::chrono::high_resolution_clock::now();
        auto cost_augmented_lagrangian_jacobian = ilqr_nodes_[index]->cost_jacobian(x, u);

        // 计算增广拉格朗日代价的二阶导数 (Hessian)
        //auto start_cost_hessian = std::chrono::high_resolution_clock::now();
        auto cost_augmented_lagrangian_hessian = ilqr_nodes_[index]->cost_hessian(x, u);
        //auto end_cost_hessian = std::chrono::high_resolution_clock::now();

        // 计算动力学的雅可比矩阵
        auto dynamics_jacobian = ilqr_nodes_[index]->dynamics_jacobian(x, u);

        // 存储所有导数信息
        cost_augmented_lagrangian_jacobian_x_list_[index] = cost_augmented_lagrangian_jacobian.first;   // ∂L_aug/∂x
        cost_augmented_lagrangian_jacobian_u_list_[index] = cost_augmented_lagrangian_jacobian.second;  // ∂L_aug/∂u
        cost_augmented_lagrangian_hessian_x_list_[index] = cost_augmented_lagrangian_hessian.first;     // ∂²L_aug/∂x²
        cost_augmented_lagrangian_hessian_u_list_[index] = cost_augmented_lagrangian_hessian.second;    // ∂²L_aug/∂u²
        dynamics_jacobian_x_list_[index] = dynamics_jacobian.first;   // ∂f/∂x (A矩阵)
        dynamics_jacobian_u_list_[index] = dynamics_jacobian.second;  // ∂f/∂u (B矩阵)

        // 计算动力学的 Hessian 张量 (用于更高精度的近似)
        dynamics_hession_x_list_[index] = ilqr_nodes_[index]->dynamics_hessian_fxx(x, u);

        // 性能分析代码 (已注释)
        // std::chrono::duration<double> cost_aug_jacobian_dur = std::chrono::duration_cast<std::chrono::duration<double>>(start_cost_hessian - start_cost_jacobian);
        // std::chrono::duration<double> cost_aug_hessian_dur = std::chrono::duration_cast<std::chrono::duration<double>>(end_cost_hessian - start_cost_hessian);
        // cost_aug_jacobian_time_sum += cost_aug_jacobian_dur.count();
        // cost_aug_hessian_time_sum +=  cost_aug_hessian_dur.count();
    }
    // std::cout << "cost_aug_jacobian_time_sum " << cost_aug_jacobian_time_sum << std::endl;
    // std::cout << "cost_aug_hessian_time_sum " << cost_aug_hessian_time_sum << std::endl;
}

template<int state_dim, int control_dim>
void NewALILQR<state_dim, control_dim>::CalcDerivatives() {
    auto x_end = x_list_.col(horizon_);
    cost_list_[horizon_] = ilqr_nodes_[horizon_]->cost(x_end, zero_control_);
    cost_augmented_lagrangian_jacobian_x_list_[horizon_] = ilqr_nodes_[horizon_]->cost_jacobian(x_end, zero_control_).first;
    cost_augmented_lagrangian_hessian_x_list_[horizon_] = ilqr_nodes_[horizon_]->cost_hessian(x_end, zero_control_).first;
    
    // auto start = std::chrono::high_resolution_clock::now();
    CalcDerivatives(0, horizon_ - 1);
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> CalcDerivatives_duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    // std::cout << "calc der " << CalcDerivatives_duration.count() << "seconds" << std::endl;
}

template<int state_dim, int control_dim>
double NewALILQR<state_dim, control_dim>::computeTotalCost() {
    return cost_list_.sum();
}

template<int state_dim, int control_dim>
void NewALILQR<state_dim, control_dim>::UpdateTrajectoryAndCostList(double alpha) {
    for(int i = 0; i < horizon_; ++i) {
        u_list_.col(i) += K_list_[i] * (x_list_.col(i) - pre_x_list_.col(i)) + alpha * k_list_[i];
        x_list_.col(i + 1) = ilqr_nodes_[i]->dynamics(x_list_.col(i), u_list_.col(i));
        cost_list_[i] = ilqr_nodes_[i]->cost(x_list_.col(i), u_list_.col(i));
    }
    cost_list_[horizon_] = ilqr_nodes_[horizon_]->cost(x_list_.col(horizon_), zero_control_);
}

template<int state_dim, int control_dim>
Eigen::Matrix<double, state_dim, PARALLEL_NUM> NewALILQR<state_dim, control_dim>::State_Dot(const Eigen::Matrix<double, state_dim, PARALLEL_NUM>& x,
                                                             const Eigen::Matrix<double, control_dim, PARALLEL_NUM>& u) {

    auto theta_list_matrix_raw = x.row(2);
    auto delta_list_matrix_raw = x.row(3);
    auto v_list_matrix_raw = x.row(4);
    auto a_list_matrix_raw = x.row(5);
    auto v_cos_theta_array = v_list_matrix_raw.array() * theta_list_matrix_raw.array().cos();
    auto v_sin_theta_array = v_list_matrix_raw.array() * theta_list_matrix_raw.array().sin();
    auto v_tan_delta_array_divide_L = v_list_matrix_raw.array() * delta_list_matrix_raw.array().tan() * (v_list_matrix_raw.Ones().array() + 0.001 * v_list_matrix_raw.array() * v_list_matrix_raw.array()).inverse() / 3.0;

    Eigen::Matrix<double, 6, PARALLEL_NUM> answer;
    answer.row(0) = v_cos_theta_array.matrix();
    answer.row(1) = v_sin_theta_array.matrix();
    answer.row(2) = v_tan_delta_array_divide_L.matrix();
    answer.row(3) = u.row(0);
    answer.row(4) = a_list_matrix_raw;
    answer.row(5) = u.row(1);
    return answer;
}

template<int state_dim, int control_dim>
void NewALILQR<state_dim, control_dim>::ParallelLinearSearch(double alpha, double& best_alpha, double& best_cost) {
    // auto start = std::chrono::high_resolution_clock::now();
    
    // Alpha preparation
    // auto alpha_prep_start = std::chrono::high_resolution_clock::now();
    auto x_list_raw = x_list_;
    auto k_forward = k_list_;
    
    Eigen::Matrix<double, PARALLEL_NUM, 1> alpha_vector;
    for (int index = 0; index < PARALLEL_NUM; ++index) {
        alpha_vector[index] = alpha;
        alpha /= 3.0;
    }
    Eigen::Array<double, control_dim, PARALLEL_NUM> real_alpha = (alpha_vector.transpose().replicate(control_dim, 1)).array();
    Eigen::Matrix<double, PARALLEL_NUM, PARALLEL_NUM> alpha_matrix = alpha_vector.asDiagonal();

    // auto alpha_prep_end = std::chrono::high_resolution_clock::now();

    // std::cout << "Alpha preparation " << " time: " 
    //              << std::chrono::duration_cast<std::chrono::microseconds>(alpha_prep_end - alpha_prep_start).count() << "us\n";
    
    // Initialization
    Eigen::Matrix<double, state_dim, PARALLEL_NUM> x_old = x_list_raw.col(0).replicate(1, PARALLEL_NUM);
    Eigen::Matrix<double, state_dim, PARALLEL_NUM> x_new = x_old;
    Eigen::Matrix<double, control_dim, PARALLEL_NUM> k_one;
    Eigen::Matrix<double, control_dim, PARALLEL_NUM> u_old;
    Eigen::Matrix<double, control_dim, PARALLEL_NUM> u_new;
    Eigen::Matrix<double, PARALLEL_NUM, 1> parallel_cost_list_;
    parallel_cost_list_.setZero();
    Eigen::Matrix<double, PARALLEL_NUM, 1> one_cost_list;



    // Main loop through the horizon
    for (int index = 0; index < horizon_; index++) {
        //auto loop_iter_start = std::chrono::high_resolution_clock::now();
        
        x_old = x_list_raw.col(index).replicate(1, PARALLEL_NUM);
        u_old = u_list_.col(index).replicate(1, PARALLEL_NUM);
        k_one = k_forward[index].replicate(1, PARALLEL_NUM);
        k_one = (k_one.array() * real_alpha).matrix();
        
        u_new = u_old + K_list_[index] * (x_new - x_old) + k_one;

        // Measure time for parallel_cost
        // auto cost_start = std::chrono::high_resolution_clock::now();
        one_cost_list = ilqr_nodes_[index]->parallel_cost(x_new, u_new);
        // auto cost_end = std::chrono::high_resolution_clock::now();


        // Measure time for parallel_dynamics
        // auto dynamics_start = std::chrono::high_resolution_clock::now();
        x_new = ilqr_nodes_[0]->parallel_dynamics(x_new, u_new);
        // auto dynamics_end = std::chrono::high_resolution_clock::now();
        
        parallel_cost_list_ += one_cost_list;

        // auto loop_iter_end = std::chrono::high_resolution_clock::now();
        // std::cout << "Total loop iteration " << index << " time: " 
        //           << std::chrono::duration_cast<std::chrono::microseconds>(loop_iter_end - loop_iter_start).count() << "us\n";
    }

    // Final cost calculation
    // auto final_cost_start = std::chrono::high_resolution_clock::now();
    one_cost_list = ilqr_nodes_[horizon_]->parallel_cost(x_new, zero_control_.replicate(1, PARALLEL_NUM));
    parallel_cost_list_ += one_cost_list;
    // auto final_cost_end = std::chrono::high_resolution_clock::now();
    // std::cout << "Final cost calculation time: " << std::chrono::duration_cast<std::chrono::microseconds>(final_cost_end - final_cost_start).count() << "us\n";

    // Determine best cost and alpha
    // auto min_cost_start = std::chrono::high_resolution_clock::now();
    Eigen::Index min_index;
    best_cost = parallel_cost_list_.minCoeff(&min_index);
    int real_index = static_cast<int>(min_index);
    best_alpha = alpha_matrix(real_index, real_index);
    // auto min_cost_end = std::chrono::high_resolution_clock::now();
    // std::cout << "Min cost calculation time: " << std::chrono::duration_cast<std::chrono::microseconds>(min_cost_end - min_cost_start).count() << "us\n";
    
    // auto end = std::chrono::high_resolution_clock::now();
    // std::cout << "Total function time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us\n";
}

/**
 * @brief Backward 函数实现 - iLQR 向后传播
 *
 * 这是 iLQR 算法的核心部分,通过动态规划从终端时刻向后传播,计算每个时间步的最优反馈控制律。
 *
 * 算法基于 Bellman 最优性原理的二次近似:
 * V(x) ≈ V_x^T x + (1/2) x^T V_xx x
 *
 * 对于每个时间步 t,计算:
 * 1. Q 函数 (action-value function) 的导数:
 *    Q_x = l_x + A^T V_x
 *    Q_u = l_u + B^T V_x
 *    Q_xx = l_xx + A^T V_xx A + Σ V_x[i] * ∂²f/∂x²[i]
 *    Q_uu = l_uu + B^T V_xx B
 *    Q_ux = B^T V_xx A
 *
 * 2. 最优控制律 (通过最小化 Q):
 *    k = -Q_uu^{-1} Q_u        (前馈项)
 *    K = -Q_uu^{-1} Q_ux       (反馈增益)
 *
 * 3. 值函数更新:
 *    V_x = Q_x + K^T Q_u + Q_ux^T k + K^T Q_uu k
 *    V_xx = Q_xx + K^T Q_uu K + Q_ux^T K + K^T Q_ux
 *
 * 4. 预测代价改善 (用于线搜索):
 *    ΔV_linear = Σ k^T Q_u
 *    ΔV_quadratic = Σ (1/2) k^T Q_uu k
 */
template<int state_dim, int control_dim>
void NewALILQR<state_dim, control_dim>::Backward() {
    // 初始化值函数的梯度和 Hessian (从终端代价开始)
    auto Vx = cost_augmented_lagrangian_jacobian_x_list_[horizon_];   // V_x = ∂L_f/∂x
    auto Vxx = cost_augmented_lagrangian_hessian_x_list_[horizon_];   // V_xx = ∂²L_f/∂x²

    // 重置预测的代价改善量
    deltaV_linear_ = 0.0;       // 线性预测: α * ΔV_linear
    deltaV_quadratic_ = 0.0;    // 二次预测: α² * ΔV_quadratic

    // 从倒数第二个时间步向后传播到第一个时间步
    for (int t = horizon_ - 1; t >= 0; --t) {
        // 获取当前时间步的动力学线性化: x_{t+1} = f(x_t, u_t) ≈ A x_t + B u_t
        auto A = dynamics_jacobian_x_list_[t];  // ∂f/∂x
        auto B = dynamics_jacobian_u_list_[t];  // ∂f/∂u

        // ========== 计算 Q 函数的导数 ==========
        // Q 函数是当前时刻代价 + 未来代价的和
        VectorControl Qu = cost_augmented_lagrangian_jacobian_u_list_[t] + B.transpose() * Vx;     // ∂Q/∂u
        VectorState Qx = cost_augmented_lagrangian_jacobian_x_list_[t] + A.transpose() * Vx;       // ∂Q/∂x
        Eigen::Matrix<double, control_dim, state_dim> Qux = B.transpose() * Vxx * A;                // ∂²Q/∂u∂x
        MatrixR Quu = cost_augmented_lagrangian_hessian_u_list_[t] + B.transpose() * Vxx * B;      // ∂²Q/∂u²
        MatrixQ Qxx = cost_augmented_lagrangian_hessian_x_list_[t] + A.transpose() * Vxx * A;      // ∂²Q/∂x²

        // 添加动力学 Hessian 项 (二阶修正): Σ V_x[i] * ∂²f_i/∂x²
        // 这提高了非线性系统的近似精度
        Qxx += std::get<0>(dynamics_hession_x_list_[t]) * Vx[0]
             + std::get<1>(dynamics_hession_x_list_[t]) * Vx[1]
             + std::get<2>(dynamics_hession_x_list_[t]) * Vx[2];

        // ========== 计算最优控制律 ==========
        MatrixR Quu_inv;
        Quu_inv = (Quu).inverse();  // 求逆 Q_uu (控制 Hessian)

        // 注: 可选的 Cholesky 分解方法 (用于数值稳定性检查,已注释)
        // auto info = Quu_chol.compute(Quu_inv).info();
        // if (info != Eigen::Success) {
        //     Vx = cost_Jx_[horizon_];
        //     Vxx = cost_Hx_[horizon_];
        //     IncreaseRegGain();  // 增加正则化
        //     break;
        // }

        MatrixK K = -Quu_inv * Qux;    // 反馈增益: K = -Q_uu^{-1} Q_ux
        VectorControl k = -Quu_inv * Qu;  // 前馈控制: k = -Q_uu^{-1} Q_u

        // 存储当前时间步的控制律
        K_list_[t] = K;
        k_list_[t] = k;

        // ========== 更新值函数 (用于下一次反向传播) ==========
        // V_x 更新 (使用 noalias() 优化性能,避免临时变量)
        Vx.noalias() = Qx + K.transpose() * (Quu * k + Qu) + Qux.transpose() * k;

        // V_xx 更新
        Vxx.noalias() = Qxx + K.transpose() * (Quu * K + Qux) + Qux.transpose() * K;

        // ========== 累积预测的代价改善 ==========
        // 用于 Forward Pass 中的线搜索判断
        deltaV_linear_ += (k.transpose() * Qu).eval()(0,0);                  // 线性项
        deltaV_quadratic_ += 0.5 * (k.transpose() * Quu * k).eval()(0,0);   // 二次项
    }
}

/**
 * @brief Forward 函数实现 - iLQR 向前传播与线搜索
 *
 * 使用 Backward Pass 计算的反馈增益 K 和前馈控制 k,通过线搜索更新轨迹。
 *
 * 控制更新策略:
 * u_new = u_old + K * (x_new - x_old) + α * k
 *
 * 其中:
 * - K: 反馈增益 (闭环控制)
 * - k: 前馈控制 (开环修正)
 * - α: 线搜索步长 (0 < α <= 1)
 *
 * 线搜索策略 (两阶段):
 * 1. 简单线搜索: α = 1, 1/2, 1/4, ...  (最多10次)
 * 2. 如果失败,调用并行线搜索: 同时评估多个步长
 *
 * 收敛判据:
 * - 如果 |ΔV_linear| < 0.2,则认为已收敛,直接返回
 */
template<int state_dim, int control_dim>
void NewALILQR<state_dim, control_dim>::Forward() {
    // 保存当前代价和轨迹 (用于比较和回退)
    double old_cost = computeTotalCost();
    double new_cost = 0.0;
    pre_x_list_ = x_list_;  // 备份状态轨迹
    pre_u_list_ = u_list_;  // 备份控制序列
    auto pre_cost_list_ = cost_list_;  // 备份代价列表

    double alpha = 1.0;       // 初始步长
    double best_alpha = 1.0;
    double best_cost = 0.0;

    // 性能分析代码 (已注释)
    // auto para_start = std::chrono::high_resolution_clock::now();
    // auto para_end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> CalcDerivatives_duration = std::chrono::duration_cast<std::chrono::duration<double>>(para_end - para_start);
    // std::cout << "parallel " << CalcDerivatives_duration.count() << "seconds" << std::endl;
    // std::cout << "best_alpha " << best_alpha << std::endl;
    // auto para_start = std::chrono::high_resolution_clock::now;

    // auto para_end = std::chrono::high_resolution_clock::now;
    // std::chrono::duration<double> para_duration = std::chrono::duration_cast<std::chrono::duration<double>>(para_end - para_start);
    // std::cout << "parallel duration " << para_duration.count() << std::endl;

    // ========== 收敛检查 ==========
    // 如果线性预测的代价改善很小,说明已接近最优,直接返回
    if (std::fabs(deltaV_linear_) < 0.2F) {
        return;
    }

    // ========== 阶段 1: 简单线搜索 ==========
    // 尝试不同的步长: 1, 1/2, 1/4, 1/8, ... (最多10次)
    for (int index = 0; index < 10; ++index) {
        // 使用当前步长 α 更新轨迹
        UpdateTrajectoryAndCostList(alpha);
        new_cost = computeTotalCost();

        // 如果新代价比旧代价小,接受这次更新
        if (new_cost < old_cost) {
            break;
        }

        // 否则,减小步长并回退轨迹
        alpha /= 2.0;
        x_list_ = pre_x_list_;
        u_list_ = pre_u_list_;
    }

    // ========== 阶段 2: 并行线搜索 (如果简单线搜索失败) ==========
    if (new_cost >= old_cost) {
        // 调用并行线搜索,同时评估多个步长
        ParallelLinearSearch(alpha, best_alpha, best_cost);

        if (best_cost >= old_cost) {
            // 并行线搜索也失败,完全回退到上一次迭代
            x_list_ = pre_x_list_;
            u_list_ = pre_u_list_;
            cost_list_ = pre_cost_list_;
        } else {
            // 并行线搜索成功,使用最优步长更新轨迹
            UpdateTrajectoryAndCostList(best_alpha);
            new_cost = best_cost;
        }
    }
}


template<int state_dim, int control_dim>
double NewALILQR<state_dim, control_dim>::ComputeConstraintViolation() {
    for (int index = 0; index < horizon_; ++index) {
        max_constraints_violation_list_(index, 0) = ilqr_nodes_[index]->max_constraints_violation(x_list_.col(index), u_list_.col(index));
    }
    max_constraints_violation_list_(horizon_, 0) = ilqr_nodes_[horizon_]->max_constraints_violation(x_list_.col(horizon_), zero_control_);
    return max_constraints_violation_list_.maxCoeff();
}

template<int state_dim, int control_dim>
void NewALILQR<state_dim, control_dim>::ILQRProcess(int max_iter, double max_tol) {
    using namespace std::chrono;
    for(int iter = 0; iter < max_iter; ++iter) {
        UpdateConstraints();


        //auto start_CalcDerivatives = high_resolution_clock::now();
        CalcDerivatives();
        //auto end_CalcDerivatives = high_resolution_clock::now();
        //duration<double> CalcDerivatives_duration = duration_cast<duration<double>>(end_CalcDerivatives - start_CalcDerivatives);
        //std::cout << "CalcDerivatives took " << CalcDerivatives_duration.count() << " seconds" << std::endl;

        double old_cost = cost_list_.sum();

        //auto start_Backward = high_resolution_clock::now();
        Backward();
        //auto end_Backward = high_resolution_clock::now();
        //<double> Backward_duration = duration_cast<duration<double>>(end_Backward - start_Backward);
        //std::cout << "Backward took " << Backward_duration.count() << " seconds" << std::endl;

        // auto start_Forward = high_resolution_clock::now();
        Forward();
        // auto end_Forward = high_resolution_clock::now();
        // duration<double> Forward_duration = duration_cast<duration<double>>(end_Forward - start_Forward);
        // std::cout << "Forward took " << Forward_duration.count() << " seconds" << std::endl;

        double new_cost = cost_list_.sum();

        if ((old_cost - new_cost < max_tol) && ((old_cost - new_cost) >= 0)) {
            break;
        }
    }
}

template<int state_dim, int control_dim>
void NewALILQR<state_dim, control_dim>::UpdateMu(double gain) {
   mu_ = mu_ * gain;
   for(auto node : ilqr_nodes_) {
      node->update_mu(mu_);
   }
}

template<int state_dim, int control_dim>
void NewALILQR<state_dim, control_dim>::UpdateLambda() {
   for(int index = 0; index < horizon_; ++index) {
      ilqr_nodes_[index]->update_lambda(x_list_.col(index), u_list_.col(index));
   }
   ilqr_nodes_[horizon_]->update_lambda(x_list_.col(horizon_), zero_control_);
}

/**
 * @brief optimize 函数实现 - 增广拉格朗日 iLQR 主优化循环
 *
 * 这是求解器的入口函数,实现增广拉格朗日法的双层迭代框架。
 *
 * @param max_outer_iter 外层最大迭代次数 (增广拉格朗日迭代)
 * @param max_inner_iter 内层最大迭代次数 (iLQR 迭代)
 * @param max_violation 约束违反容差 (收敛判据)
 *
 * 算法框架 (增广拉格朗日法 - Augmented Lagrangian Method):
 *
 * 1. 初始化: linearizedInitialGuess()
 *    使用 LQR 生成初始轨迹
 *
 * 2. 外层循环 (约束处理):
 *    for k = 1 to max_outer_iter:
 *      a) 内层 iLQR 优化: 最小化增广拉格朗日函数
 *         L_aug(x, u) = L(x, u) + λ^T c(x, u) + (μ/2) ||c(x, u)||²
 *
 *      b) 计算约束违反量: max_violation = max |c(x, u)|
 *
 *      c) 检查收敛:
 *         if max_violation < tolerance:
 *             收敛退出
 *
 *      d) 更新增广拉格朗日参数:
 *         - 如果 max_violation > 5 * tolerance: (约束违反严重)
 *             大幅增加惩罚系数: μ = μ * 100
 *         - 否则: (约束违反适中)
 *             更新拉格朗日乘子: λ = λ + μ * c(x, u)
 *
 * 增广拉格朗日法的优势:
 * - 相比纯惩罚法,不需要 μ → ∞,数值条件更好
 * - 相比拉格朗日法,对初始猜测不敏感,鲁棒性更强
 * - 能够处理不等式约束: c(x, u) <= 0
 */
template<int state_dim, int control_dim>
void NewALILQR<state_dim, control_dim>::optimize(int max_outer_iter, int max_inner_iter, double max_violation) {
    using namespace std::chrono;
    auto start_optimize = high_resolution_clock::now();

    // ========== 初始化: 生成初始轨迹猜测 ==========
    linearizedInitialGuess();

    // ========== 外层循环: 增广拉格朗日迭代 ==========
    for (int index = 0; index < max_outer_iter; ++index) {
        // ===== 内层优化: iLQR 最小化增广拉格朗日函数 =====
        // auto start_ILQR = high_resolution_clock::now();
        ILQRProcess(max_inner_iter, 1e-3);  // 最多 max_inner_iter 次 iLQR 迭代
        // auto end_ILQR = high_resolution_clock::now();
        // duration<double> ILQR_duration = duration_cast<duration<double>>(end_ILQR - start_ILQR);
        // std::cout << "ILQRProcess took " << ILQR_duration.count() << " seconds" << std::endl;

        // ===== 计算当前轨迹的约束违反量 =====
        double inner_violation = ComputeConstraintViolation();
        // std::cout << "inner_violation" << inner_violation << std::endl;

        // ===== 检查收敛条件 =====
        if (inner_violation < max_violation) {
            // 所有约束满足容差,优化成功收敛
            break;
        } else {
            // ===== 更新增广拉格朗日参数 =====
            if (inner_violation > 5 * max_violation) {
                // 约束违反严重: 大幅增加惩罚系数 μ
                // 这会在下一次 iLQR 迭代中强制约束满足
                UpdateMu(100.0);  // μ_new = μ_old * 100
            } else {
                // 约束违反适中: 更新拉格朗日乘子 λ
                // 这是增广拉格朗日法的核心步骤
                UpdateLambda();   // λ_new = λ_old + μ * c(x, u)
            }
        }
    }

    // ========== 性能统计 ==========
    auto end_optimize = high_resolution_clock::now();
    duration<double> optimize_duration = duration_cast<duration<double>>(end_optimize - start_optimize);

    std::cout << "optimize took " << optimize_duration.count() << " seconds" << std::endl;
}

#endif // NEW_ALILQR_H
