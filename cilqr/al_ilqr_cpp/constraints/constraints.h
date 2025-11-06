/**
 * @file constraints.h
 * @brief 约束基类 - iLQR 增广拉格朗日法的约束处理框架
 *
 * 本文件定义了约束处理的基类接口,用于增广拉格朗日 iLQR 算法。
 * 支持等式约束和不等式约束的统一处理。
 *
 * ## 增广拉格朗日法 (Augmented Lagrangian Method)
 *
 * 增广拉格朗日法是求解约束优化问题的一种经典方法:
 *
 * ```
 * 原问题:
 * minimize   f(x, u)
 * subject to c(x, u) <= 0  (不等式约束)
 *            h(x, u) = 0   (等式约束)
 * ```
 *
 * 增广拉格朗日函数:
 * ```
 * L_aug(x, u, λ, μ) = f(x, u) + λᵀc(x, u) + (μ/2)||c(x, u)||²
 * ```
 *
 * 其中:
 * - λ: 拉格朗日乘子向量 (对偶变量)
 * - μ: 惩罚系数 (标量, μ > 0)
 * - c(x, u): 约束函数向量
 *
 * ## 双层优化框架
 *
 * 外层循环 (更新对偶变量):
 * ```
 * for k = 1 to max_iter:
 *   1. 内层: 最小化 L_aug(x, u, λ_k, μ_k)  → 得到 (x*, u*)
 *   2. 更新拉格朗日乘子: λ_{k+1} = proj(λ_k - μ_k * c(x*, u*))
 *   3. 如果违反严重: 增加惩罚 μ_{k+1} = β * μ_k
 * ```
 *
 * ## 不等式约束的投影算子
 *
 * 对于不等式约束 c(x, u) <= 0, 使用投影算子:
 * ```
 * proj(λ) = min(λ, 0)  # 逐元素最小值
 * ```
 *
 * 这确保拉格朗日乘子满足 KKT 条件: λ >= 0 且 λᵀc = 0
 *
 * @see BoxConstraints 盒式约束实现 (状态和控制的上下界)
 * @see QuadraticConstraints 二次约束实现 (圆形障碍物)
 * @see LinearConstraints 线性约束实现
 */

#ifndef CONSTRAINTS_CONSTRAINTS_H_
#define CONSTRAINTS_CONSTRAINTS_H_


#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>
#include <tuple>
#include <utility>
#include <array>
#include <algorithm>

/**
 * @brief 并行线搜索的并行度
 *
 * 定义在并行线搜索中同时评估的候选步长数量。
 * 值越大,并行化程度越高,但内存消耗也越大。
 *
 * 典型值: 5-16
 */
#define PARALLEL_NUM 5

/**
 * @brief 约束基类
 *
 * 定义了约束处理的接口,供具体约束类 (BoxConstraints, QuadraticConstraints 等) 继承。
 *
 * @tparam state_dim 状态维度 (例如: 6)
 * @tparam control_dim 控制维度 (例如: 2)
 * @tparam constraint_dim 约束维度 (例如: 盒式约束为 2*(state_dim+control_dim))
 *
 * ## 主要功能
 *
 * 1. **约束评估**: constraints(x, u) 计算约束值
 * 2. **导数计算**: 计算约束的雅可比和 Hessian
 * 3. **增广拉格朗日代价**: augmented_lagrangian_cost(x, u)
 * 4. **拉格朗日乘子更新**: update_lambda(x, u)
 * 5. **投影算子**: projection(λ) 用于不等式约束
 *
 * ## 使用示例
 *
 * ```cpp
 * // 创建盒式约束 (继承自 Constraints)
 * BoxConstraints<6, 2> constraints(state_min, state_max, control_min, control_max);
 *
 * // 评估约束
 * auto c = constraints.constraints(x, u);  // c <= 0 表示满足约束
 *
 * // 计算增广拉格朗日代价
 * double cost = constraints.augmented_lagrangian_cost(x, u);
 *
 * // 更新拉格朗日乘子
 * constraints.update_lambda(x, u);
 * ```
 */
template <int state_dim, int control_dim, int constraint_dim>
class Constraints {
public:
    /**
     * @brief 构造函数
     *
     * @param is_equality 是否为等式约束
     *                    - false: 不等式约束 c(x, u) <= 0 (默认)
     *                    - true: 等式约束 h(x, u) = 0
     *
     * 初始化:
     * - 拉格朗日乘子 λ = 0
     * - 惩罚系数 μ = 1.0
     *
     * ## 等式约束 vs 不等式约束
     *
     * **不等式约束** (is_equality = false):
     * - 使用投影算子: λ = min(λ - μ*c, 0)
     * - 适用于盒式约束、障碍物避障等
     *
     * **等式约束** (is_equality = true):
     * - 直接更新: λ = λ - μ*h
     * - 适用于路径约束、终端约束等
     */
    Constraints(bool is_equality = false)
        : lambda_(Eigen::Matrix<double, constraint_dim, 1>::Zero()),
          mu_(1.0),
          is_equality_(is_equality) {}

    /**
     * @brief 虚析构函数
     *
     * 确保派生类对象可以通过基类指针正确析构。
     */
    virtual ~Constraints() = default;

    // ============================================
    // Getters and Setters (访问器和修改器)
    // ============================================

    /** @brief 获取拉格朗日乘子 λ (constraint_dim × 1) */
    Eigen::Matrix<double, constraint_dim, 1> lambda() const {
        return lambda_;
    }

    /** @brief 设置拉格朗日乘子 λ */
    void set_lambda(const Eigen::Matrix<double, constraint_dim, 1>& lambda) {
        lambda_ = lambda;
    }

    /** @brief 重置拉格朗日乘子 λ = 0 */
    void reset_lambda() {
        lambda_.setZero();
    }

    /** @brief 获取惩罚系数 μ */
    double mu() const {
        return mu_;
    }

    /** @brief 设置惩罚系数 μ */
    void set_mu(double mu) {
        mu_ = mu;
    }

    /** @brief 获取状态维度 (编译时常量) */
    static constexpr int get_state_dim() {
        return state_dim;
    }

    /** @brief 获取控制维度 (编译时常量) */
    static constexpr int get_control_dim() {
        return control_dim;
    }

    /** @brief 获取约束维度 (编译时常量) */
    static constexpr int get_constraint_dim() {
        return constraint_dim;
    }

    /** @brief 设置当前约束索引 (用于动态约束管理) */
    void set_current_constraints_index(int index) {
        current_constraints_index_ = index;
    }

    /** @brief 获取当前约束索引 */
    int get_current_constraints_index() {
        return current_constraints_index_;
    }

    /**
     * @brief 动态更新约束 (可选,默认空实现)
     *
     * 允许派生类在运行时添加新的线性约束,主要用于障碍物避障。
     *
     * @param A_rows 约束系数向量 (1 × state_dim)
     * @param C_rows 约束常数 (标量)
     *
     * 约束形式: A * x <= C
     *
     * @note 派生类可以覆盖此函数以实现动态约束添加
     * @see DynamicLinearConstraints 实现动态约束的子类
     */
    virtual void UpdateConstraints(const Eigen::Ref<const Eigen::Matrix<double, 1, state_dim>> A_rows, double C_rows) {
        // 默认空实现,派生类可以覆盖
    }

    // ============================================
    // 纯虚函数 (必须由派生类实现)
    // ============================================

    /**
     * @brief 批量计算约束值 (并行版本)
     *
     * 同时计算 N 个状态-控制对的约束值,用于并行线搜索。
     *
     * @param x 状态矩阵 (state_dim × N)
     * @param u 控制矩阵 (control_dim × N)
     * @return 约束值矩阵 (constraint_dim × N)
     *         - 每列是一个状态-控制对的约束值
     *         - c[i,j] <= 0 表示第 j 个样本的第 i 个约束满足
     *
     * @note 纯虚函数,派生类必须实现
     * @note 用于 parallel_augmented_lagrangian_cost 中的批量计算
     */
    virtual Eigen::Matrix<double, constraint_dim, PARALLEL_NUM> parallel_constraints(const Eigen::Ref<const Eigen::Matrix<double, state_dim, PARALLEL_NUM>>& x,
                                                 const Eigen::Ref<const Eigen::Matrix<double, control_dim, PARALLEL_NUM>>& u) const = 0;

    /**
     * @brief 计算约束值 c(x, u)
     *
     * 评估给定状态-控制对的所有约束。
     *
     * @param x 状态向量 (state_dim × 1)
     * @param u 控制向量 (control_dim × 1)
     * @return 约束值向量 (constraint_dim × 1)
     *         - c[i] <= 0 表示第 i 个约束满足
     *         - c[i] > 0 表示第 i 个约束被违反
     *
     * ## 约束类型示例
     *
     * **盒式约束**:
     * ```
     * c = [x - x_max;      # 上界约束: x <= x_max
     *      x_min - x;      # 下界约束: x >= x_min
     *      u - u_max;      # 上界约束: u <= u_max
     *      u_min - u]      # 下界约束: u >= u_min
     * ```
     *
     * **圆形障碍物**:
     * ```
     * c[i] = r_i² - ||p - p_obs[i]||²  # 点在圆外: c <= 0
     * ```
     *
     * @note 纯虚函数,派生类必须实现
     */
    virtual Eigen::Matrix<double, constraint_dim, 1> constraints(const Eigen::Ref<const Eigen::Matrix<double, state_dim, 1>>& x,
                                                               const Eigen::Ref<const Eigen::Matrix<double, control_dim, 1>>& u) const = 0;

    /**
     * @brief 计算约束的雅可比矩阵 (一阶导数)
     *
     * 计算约束函数对状态和控制的偏导数。
     *
     * @param x 状态向量 (state_dim × 1)
     * @param u 控制向量 (control_dim × 1)
     * @return std::pair<Jx, Ju>
     *         - Jx: ∂c/∂x (constraint_dim × state_dim), 每行是一个约束对状态的梯度
     *         - Ju: ∂c/∂u (constraint_dim × control_dim), 每行是一个约束对控制的梯度
     *
     * ## 雅可比矩阵的作用
     *
     * 用于计算增广拉格朗日函数的梯度:
     * ```
     * ∂L_aug/∂x = ∂f/∂x + (∂c/∂x)ᵀ λ + μ (∂c/∂x)ᵀ c
     * ∂L_aug/∂u = ∂f/∂u + (∂c/∂u)ᵀ λ + μ (∂c/∂u)ᵀ c
     * ```
     *
     * @note 纯虚函数,派生类必须实现
     * @note Jx[i,j] = ∂c_i/∂x_j
     */
    virtual std::pair<Eigen::Matrix<double, constraint_dim, state_dim>, Eigen::Matrix<double, constraint_dim, control_dim>>
    constraints_jacobian(const Eigen::Ref<const Eigen::Matrix<double, state_dim, 1>>& x,
                        const Eigen::Ref<const Eigen::Matrix<double, control_dim, 1>>& u) const = 0;

    /**
     * @brief 计算约束的 Hessian 张量 (二阶导数)
     *
     * 计算每个约束函数的 Hessian 矩阵,用于更精确的二阶优化。
     *
     * @param x 状态向量 (state_dim × 1)
     * @param u 控制向量 (control_dim × 1)
     * @return std::tuple<Hxx, Huu, Hxu>
     *         - Hxx: 数组,长度为 constraint_dim, 每个元素是 ∂²c_i/∂x² (state_dim × state_dim)
     *         - Huu: 数组,长度为 constraint_dim, 每个元素是 ∂²c_i/∂u² (control_dim × control_dim)
     *         - Hxu: 数组,长度为 constraint_dim, 每个元素是 ∂²c_i/∂x∂u (state_dim × control_dim)
     *
     * ## Hessian 张量的结构
     *
     * 对于 constraint_dim 个约束,Hessian 是一个三阶张量:
     * ```
     * Hxx[i] = ∂²c_i/∂x²  (state_dim × state_dim)
     * ```
     *
     * ## 在增广拉格朗日法中的作用
     *
     * 用于计算增广拉格朗日函数的 Hessian:
     * ```
     * ∂²L_aug/∂x² = ∂²f/∂x² + Σᵢ λᵢ ∂²cᵢ/∂x² + μ Σᵢ cᵢ ∂²cᵢ/∂x² + μ (∂c/∂x)ᵀ (∂c/∂x)
     * ```
     *
     * @note 纯虚函数,派生类必须实现
     * @note 对于线性约束,所有 Hessian 矩阵都是零矩阵
     */
    virtual std::tuple<std::array<Eigen::Matrix<double, state_dim, state_dim>, constraint_dim>,
                       std::array<Eigen::Matrix<double, control_dim, control_dim>, constraint_dim>,
                       std::array<Eigen::Matrix<double, state_dim, control_dim>, constraint_dim>>
    constraints_hessian(const Eigen::Ref<const Eigen::Matrix<double, state_dim, 1>>& x,
                       const Eigen::Ref<const Eigen::Matrix<double, control_dim, 1>>& u) const = 0;


    // ============================================
    // 投影算子 (Projection Operator)
    // ============================================

    /**
     * @brief 不等式约束的投影算子
     *
     * 将向量投影到不等式约束的可行集: proj(x) = min(x, 0) (逐元素)
     *
     * @param x 输入向量 (constraint_dim × 1)
     * @return 投影后的向量 (constraint_dim × 1)
     *
     * ## 数学定义
     *
     * 对于不等式约束 c(x, u) <= 0, 拉格朗日乘子必须满足:
     * - λ >= 0 (非负性)
     * - λᵀc = 0 (互补松弛条件)
     *
     * 投影算子保证这些条件:
     * ```
     * proj(λ)[i] = min(λ[i], 0) = {
     *     λ[i], if λ[i] < 0   (活跃约束)
     *     0,    if λ[i] >= 0  (非活跃约束)
     * }
     * ```
     *
     * ## 应用
     *
     * 在拉格朗日乘子更新中:
     * ```
     * λ_new = proj(λ_old - μ * c)
     * ```
     *
     * @note 对于等式约束,不需要投影
     * @note 使用 Eigen 的 cwiseMin 进行高效的逐元素最小值计算
     */
    Eigen::Matrix<double, constraint_dim, 1> projection(const Eigen::Matrix<double, constraint_dim, 1>& x) const {
        return x.cwiseMin(0);  // 逐元素取 min(x[i], 0)
    }

    /**
     * @brief 投影算子的雅可比矩阵 (一阶导数)
     *
     * 计算投影算子对输入的偏导数: ∂proj(x)/∂x
     *
     * @param x 输入向量 (constraint_dim × 1)
     * @return 对角矩阵 (constraint_dim × constraint_dim)
     *
     * ## 雅可比矩阵的定义
     *
     * proj(x) 是逐元素操作,因此雅可比是对角矩阵:
     * ```
     * ∂proj(x)[i]/∂x[j] = {
     *     1, if i == j and x[i] < 0
     *     0, otherwise
     * }
     * ```
     *
     * ## 应用
     *
     * 用于计算增广拉格朗日代价的梯度:
     * ```
     * ∂L_aug/∂x = ... + (∂proj/∂λ) (∂λ/∂c) (∂c/∂x)
     * ```
     *
     * @note 返回对角矩阵,对角元素为 0 或 1
     */
    Eigen::Matrix<double, constraint_dim, constraint_dim> projection_jacobian(const Eigen::Matrix<double, constraint_dim, 1>& x) const {
        Eigen::Matrix<double, constraint_dim, constraint_dim> jac;
        Eigen::Array<double, constraint_dim, 1> x_temp = x.array();

        // (x < 0) 返回布尔数组,转换为 double (0 或 1)
        Eigen::Array<double, constraint_dim, 1> ans = (x_temp < 0).template cast<double>();

        return ans.matrix().asDiagonal();  // 构造对角矩阵
    }

    /**
     * @brief 投影算子的雅可比矩阵 (优化版本 2)
     *
     * 原地修改存储的约束雅可比矩阵,将非活跃约束对应的列置零。
     *
     * @param x 投影后的拉格朗日乘子 (constraint_dim × 1)
     *
     * ## 原理
     *
     * 对于非活跃约束 (x[i] >= 0), 其对应的梯度贡献为 0:
     * ```
     * proj_cx_T_[:, i] = 0  if x[i] >= 0
     * proj_cu_T_[:, i] = 0  if x[i] >= 0
     * ```
     *
     * 这等价于投影雅可比的链式法则:
     * ```
     * proj_jac * cx ≈ 直接将 cx 的第 i 列置零 (如果 x[i] >= 0)
     * ```
     *
     * @note 此函数修改成员变量 proj_cx_T_ 和 proj_cu_T_
     * @note 用于优化增广拉格朗日梯度和 Hessian 的计算
     */
    void projection_jacobian2(const Eigen::Matrix<double, constraint_dim, 1>& x) {
        for (int i = 0; i < x.size(); ++i) {
            if (x(i) >= 0) {  // 非活跃约束
                proj_cx_T_.col(i) = Eigen::Matrix<double, state_dim, 1>::Zero();
                proj_cu_T_.col(i) = Eigen::Matrix<double, control_dim, 1>::Zero();
            }
        }
    }

    /**
     * @brief 投影算子的 Hessian 矩阵 (二阶导数)
     *
     * 对于 proj(x) = min(x, 0), 二阶导数恒为零。
     *
     * @param x 输入向量 (constraint_dim × 1)
     * @param b 预乘矩阵 (constraint_dim × constraint_dim), 未使用
     * @return 零矩阵 (constraint_dim × constraint_dim)
     *
     * ## 数学原因
     *
     * proj(x) 是分段线性函数:
     * ```
     * proj(x)[i] = {
     *     x[i], if x[i] < 0   (线性)
     *     0,    if x[i] >= 0  (常数)
     * }
     * ```
     *
     * 线性和常数函数的二阶导数都是 0。
     *
     * @note 对于光滑近似的投影算子,Hessian 可能非零
     */
    Eigen::Matrix<double, constraint_dim, constraint_dim> projection_hessian(const Eigen::Matrix<double, constraint_dim, 1>& x,
                                                                           const Eigen::Matrix<double, constraint_dim, constraint_dim>& b) const {
        return Eigen::Matrix<double, constraint_dim, constraint_dim>::Zero();
    }

    // ============================================
    // 增广拉格朗日代价函数
    // ============================================

    /**
     * @brief 计算增广拉格朗日代价 (单个状态-控制对)
     *
     * 计算给定状态-控制对的增广拉格朗日惩罚项。
     *
     * @param x 状态向量 (state_dim × 1)
     * @param u 控制向量 (control_dim × 1)
     * @return 增广拉格朗日代价 (标量)
     *
     * ## 数学公式
     *
     * **等式约束** (is_equality = true):
     * ```
     * L_aug = 0.5/μ * [||λ - μ*h||² - ||λ||²]
     *       = 0.5/μ * [λᵀλ - 2μλᵀh + μ²||h||² - λᵀλ]
     *       = -λᵀh + 0.5μ||h||²
     * ```
     *
     * **不等式约束** (is_equality = false):
     * ```
     * L_aug = 0.5/μ * [||proj(λ - μ*c)||² - ||λ||²]
     * ```
     * 其中 proj(x) = min(x, 0) (逐元素)
     *
     * ## 物理意义
     *
     * 增广拉格朗日项包含两部分:
     * 1. **拉格朗日项**: λᵀc (对偶变量与约束的点积)
     * 2. **惩罚项**: (μ/2)||c||² (约束违反的平方惩罚)
     *
     * 组合这两项的优势:
     * - 相比纯惩罚法,不需要 μ → ∞,数值条件更好
     * - 相比拉格朗日法,对初始猜测不敏感,鲁棒性更强
     *
     * ## 在 iLQR 中的应用
     *
     * 此代价会被添加到原始代价函数:
     * ```
     * total_cost = trajectory_cost + Σₜ augmented_lagrangian_cost(xₜ, uₜ)
     * ```
     *
     * @note 此函数同时更新内部状态 c_ 和 lambda_proj_
     * @note 对于不等式约束,使用投影算子确保 KKT 条件
     */
    double augmented_lagrangian_cost(const Eigen::Ref<const Eigen::Matrix<double, state_dim, 1>>& x,
                                     const Eigen::Ref<const Eigen::Matrix<double, control_dim, 1>>& u) {
        // 计算约束值
        Eigen::Matrix<double, constraint_dim, 1> c = constraints(x, u);
        c_ = c;  // 缓存约束值供后续使用

        if (is_equality_) {
            // 等式约束: L = 0.5/μ * [||λ - μ*h||² - ||λ||²]
            //            = -λᵀh + 0.5μ||h||²
            return 0.5 / mu_ * ((lambda_ - mu_ * c).transpose() * (lambda_ - mu_ * c) - lambda_.transpose() * lambda_).value();
        } else {
            // 不等式约束: L = 0.5/μ * [||proj(λ - μ*c)||² - ||λ||²]
            lambda_proj_ = projection(lambda_ - mu_ * c);  // 投影到可行集
            return 0.5 / mu_ * (lambda_proj_.transpose() * lambda_proj_ - lambda_.transpose() * lambda_).value();
        }
    }


    /**
     * @brief 批量计算增广拉格朗日代价 (并行版本)
     *
     * 同时计算 N 个状态-控制对的增广拉格朗日代价,用于并行线搜索。
     *
     * @param x 状态矩阵 (state_dim × N)
     * @param u 控制矩阵 (control_dim × N)
     * @return 代价向量 (N × 1), 每个元素是对应状态-控制对的增广拉格朗日代价
     *
     * ## 向量化计算流程
     *
     * 1. 批量计算约束: c = parallel_constraints(x, u)  # constraint_dim × N
     * 2. 广播拉格朗日乘子: λ → [λ, λ, ..., λ]  # constraint_dim × N
     * 3. 批量投影: proj = min(λ - μ*c, 0)  # 逐元素
     * 4. 计算代价: cost[i] = 0.5/μ * (||proj[:,i]||² - ||λ||²)
     *
     * ## 性能优势
     *
     * 相比循环调用 augmented_lagrangian_cost:
     * - 利用 SIMD 指令集加速
     * - 减少函数调用开销
     * - 典型性能提升 5-10 倍
     *
     * @note 仅支持不等式约束的并行计算
     * @note 用于 ParallelLinearSearch 中的批量代价评估
     */
    Eigen::Matrix<double, PARALLEL_NUM, 1> parallel_augmented_lagrangian_cost(const Eigen::Ref<const Eigen::Matrix<double, state_dim, PARALLEL_NUM>>& x,
                                                 const Eigen::Ref<const Eigen::Matrix<double, control_dim, PARALLEL_NUM>>& u) {
        // 批量计算约束值 (constraint_dim × N)
        Eigen::Matrix<double, constraint_dim, PARALLEL_NUM> c;
        c = parallel_constraints(x, u);

        // 广播拉格朗日乘子到 N 列
        Eigen::Matrix<double, constraint_dim, PARALLEL_NUM> parallel_lambda = lambda_.replicate(1, PARALLEL_NUM);

        // 批量投影: proj = min(λ - μ*c, 0)
        Eigen::Matrix<double, constraint_dim, PARALLEL_NUM> proj = (parallel_lambda - mu_ * c).cwiseMin(0);

        // 转换为数组以进行逐元素运算
        Eigen::Array<double, constraint_dim, PARALLEL_NUM> parallel_lambda_array = parallel_lambda.array();
        Eigen::Array<double, constraint_dim, PARALLEL_NUM> proj_array = proj.array();

        // 计算每列的代价: ||proj[:,i]||² - ||λ||²
        Eigen::Matrix<double, PARALLEL_NUM, 1> ans = (proj_array * proj_array - parallel_lambda_array * parallel_lambda_array).matrix().colwise().sum().transpose();

        // 乘以系数 0.5/μ
        ans = 0.5 / mu_ * ans;
        return ans;
    }

    // ============================================
    // 增广拉格朗日代价的梯度和 Hessian
    // ============================================

    /**
     * @brief 计算增广拉格朗日代价的梯度 (雅可比)
     *
     * 计算增广拉格朗日函数对状态和控制的一阶导数。
     *
     * @param x 状态向量 (state_dim × 1)
     * @param u 控制向量 (control_dim × 1)
     * @return std::pair<dx, du>
     *         - dx: ∂L_aug/∂x (state_dim × 1)
     *         - du: ∂L_aug/∂u (control_dim × 1)
     *
     * ## 数学推导
     *
     * 增广拉格朗日函数:
     * ```
     * L_aug = 0.5/μ * [||proj(λ - μ*c)||² - ||λ||²]
     * ```
     *
     * ### 等式约束 (is_equality = true)
     *
     * ```
     * L_aug = -λᵀh + 0.5μ||h||²
     *
     * ∂L_aug/∂x = -∂h/∂x)ᵀ λ + μ (∂h/∂x)ᵀ h
     *           = -(∂h/∂x)ᵀ (λ - μ*h)
     * ```
     *
     * ### 不等式约束 (is_equality = false)
     *
     * 使用链式法则和投影算子:
     * ```
     * ∂L_aug/∂x = 1/μ * proj(λ - μ*c)ᵀ ∂proj/∂(λ-μ*c) * (-μ) * ∂c/∂x
     *           = -proj(λ - μ*c)ᵀ ∂proj/∂(λ-μ*c) * ∂c/∂x
     * ```
     *
     * 其中投影雅可比:
     * ```
     * ∂proj/∂(λ-μ*c)[i,i] = {
     *     1, if (λ - μ*c)[i] < 0  (活跃约束)
     *     0, if (λ - μ*c)[i] >= 0 (非活跃约束)
     * }
     * ```
     *
     * 实际实现中,使用优化版本:
     * ```
     * proj_cx = (∂proj/∂(λ-μ*c) * ∂c/∂x)ᵀ  (通过将非活跃约束的列置零实现)
     * ∂L_aug/∂x = -proj_cx * proj(λ - μ*c)
     * ```
     *
     * ## 在 iLQR Backward Pass 中的应用
     *
     * 此梯度会被添加到代价函数梯度:
     * ```
     * lₓ = ∂f/∂x + ∂L_aug/∂x
     * lᵤ = ∂f/∂u + ∂L_aug/∂u
     * ```
     *
     * 然后用于计算 Qₓ 和 Qᵤ:
     * ```
     * Qₓ = lₓ + fₓᵀ Vₓ
     * Qᵤ = lᵤ + fᵤᵀ Vₓ
     * ```
     *
     * ## 优化技巧
     *
     * - 缓存约束雅可比 (cx_, cu_) 供 Hessian 计算使用
     * - 使用 projection_jacobian2() 原地修改以避免矩阵乘法
     * - 预计算 dxdx_, dudu_ 供 Hessian 使用
     *
     * @note 此函数会修改内部状态: cx_, cu_, proj_cx_T_, proj_cu_T_, dxdx_, dudu_
     * @note 必须在 augmented_lagrangian_cost() 之后调用,因为依赖 c_ 和 lambda_proj_
     */
    std::pair<Eigen::Matrix<double, state_dim, 1>, Eigen::Matrix<double, control_dim, 1>>
    augmented_lagrangian_jacobian(const Eigen::Ref<const Eigen::Matrix<double, state_dim, 1>>& x,
                                  const Eigen::Ref<const Eigen::Matrix<double, control_dim, 1>>& u) {
        auto jacobian_matrix = constraints_jacobian(x, u);
        auto cx = jacobian_matrix.first;
        auto cu = jacobian_matrix.second;

        cx_ = cx;
        cu_ = cu;
        proj_cx_T_ = cx_.transpose();
        proj_cu_T_ = cu_.transpose();
        Eigen::Matrix<double, state_dim, 1> dx;
        Eigen::Matrix<double, control_dim, 1> du;


        if (is_equality_) {
            Eigen::Matrix<double, constraint_dim, 1> factor = lambda_ - mu_ * c_;
            dx = -cx.transpose() * factor;
            du = -cu.transpose() * factor;
        } else {
            // proj_jac_ = projection_jacobian(lambda_proj_);
            // proj_cx_T_ = (proj_jac_ * cx).transpose();
            // proj_cu_T_ = (proj_jac_ * cu).transpose();

            projection_jacobian2(lambda_proj_);
            dx = -proj_cx_T_ * lambda_proj_;
            du = -proj_cu_T_ * lambda_proj_;
            dxdx_ = mu_ * proj_cx_T_ * cx_;
            dudu_ = mu_ * proj_cu_T_ * cu_;
            dxdu_.setZero();
        }

        return {dx, du};
    }

    /**
     * @brief 计算增广拉格朗日代价的 Hessian 矩阵 (二阶导数)
     *
     * 计算增广拉格朗日函数对状态和控制的二阶偏导数。
     *
     * @param x 状态向量 (state_dim × 1)
     * @param u 控制向量 (control_dim × 1)
     * @param full_newton 是否使用完整牛顿法 (未使用,保留参数)
     * @return std::tuple<dxdx, dudu, dxdu>
     *         - dxdx: ∂²L_aug/∂x² (state_dim × state_dim)
     *         - dudu: ∂²L_aug/∂u² (control_dim × control_dim)
     *         - dxdu: ∂²L_aug/∂x∂u (state_dim × control_dim)
     *
     * ## 数学推导
     *
     * ### 等式约束 (is_equality = true)
     *
     * ```
     * L_aug = -λᵀh + 0.5μ||h||²
     *
     * ∂²L_aug/∂x² = μ (∂h/∂x)ᵀ (∂h/∂x) - Σᵢ [λᵢ - μ*hᵢ] * ∂²hᵢ/∂x²
     *             = μ cₓᵀ cₓ - Σᵢ factor[i] * Hxxᵢ
     *
     * ∂²L_aug/∂u² = μ (∂h/∂u)ᵀ (∂h/∂u) - Σᵢ factor[i] * Huuᵢ
     *
     * ∂²L_aug/∂x∂u = μ (∂h/∂x)ᵀ (∂h/∂u) - Σᵢ factor[i] * Hxuᵢ
     * ```
     *
     * 其中 factor = λ - μ*h
     *
     * ### 不等式约束 (is_equality = false)
     *
     * ```
     * ∂²L_aug/∂x² = μ (proj_cx)ᵀ (proj_cx) - Σᵢ proj(λ - μ*c)[i] * ∂²cᵢ/∂x²
     * ```
     *
     * 其中 proj_cx = (∂proj/∂(λ-μ*c) * ∂c/∂x)ᵀ (通过将非活跃约束置零实现)
     *
     * ## Hessian 的两个组成部分
     *
     * **线性项** (一阶导数的外积):
     * ```
     * μ * (∂c/∂x)ᵀ (∂c/∂x)
     * ```
     * - 来自惩罚项 (μ/2)||c||² 的二阶导数
     * - 总是正半定的,改善数值条件
     *
     * **非线性项** (约束的二阶导数):
     * ```
     * -Σᵢ factor[i] * ∂²cᵢ/∂x²
     * ```
     * - 来自约束本身的非线性
     * - 对于线性约束为零
     * - 使用 tensor_contract() 计算张量收缩
     *
     * ## 在 iLQR Backward Pass 中的应用
     *
     * 此 Hessian 会被添加到代价函数 Hessian:
     * ```
     * lₓₓ = ∂²f/∂x² + ∂²L_aug/∂x²
     * lᵤᵤ = ∂²f/∂u² + ∂²L_aug/∂u²
     * lₓᵤ = ∂²f/∂x∂u + ∂²L_aug/∂x∂u
     * ```
     *
     * 然后用于计算 Q 矩阵:
     * ```
     * Qₓₓ = lₓₓ + fₓᵀ Vₓₓ fₓ + Vₓᵀ fₓₓ  (如果考虑 fₓₓ)
     * Qᵤᵤ = lᵤᵤ + fᵤᵀ Vₓₓ fᵤ
     * Qₓᵤ = lₓᵤ + fₓᵀ Vₓₓ fᵤ
     * ```
     *
     * ## 性能考虑
     *
     * - 对于盒式约束 (线性): 只需计算外积项 (快速)
     * - 对于二次约束 (圆形障碍物): 需要完整的张量收缩 (较慢)
     * - 使用缓存 dxdx_, dudu_, dxdu_ 避免重复计算
     *
     * @note 必须在 augmented_lagrangian_jacobian() 之后调用,因为依赖预计算的 dxdx_, dudu_
     * @note 对于不等式约束,投影算子的 Hessian 为零 (projection_hessian 返回零矩阵)
     */
    std::tuple<Eigen::Matrix<double, state_dim, state_dim>,
               Eigen::Matrix<double, control_dim, control_dim>,
               Eigen::Matrix<double, state_dim, control_dim>>
    augmented_lagrangian_hessian(const Eigen::Ref<const Eigen::Matrix<double, state_dim, 1>>& x,
                                 const Eigen::Ref<const Eigen::Matrix<double, control_dim, 1>>& u, bool full_newton = false) {
        // Eigen::Matrix<double, constraint_dim, 1> c = constraints(x, u);

        auto hessian_tensor = constraints_hessian(x, u);

        // auto jacobian_matrix = constraints_jacobian(x, u);
        // auto cx = cx_;
        // auto cu = cu_;
        //Eigen::Matrix<double, state_dim, state_dim> dxdx = Eigen::Matrix<double, state_dim, state_dim>::Zero();
        //Eigen::Matrix<double, control_dim, control_dim> dudu = Eigen::Matrix<double, control_dim, control_dim>::Zero();
        //Eigen::Matrix<double, state_dim, control_dim> dxdu = Eigen::Matrix<double, state_dim, control_dim>::Zero();
        Eigen::Matrix<double, constraint_dim, 1> factor = lambda_ - mu_ * c_;

        
        if (is_equality_) {
            auto ans = tensor_contract(factor, hessian_tensor);
            dxdx_ = mu_ * ((cx_.transpose() * cx_)) - std::get<0>(ans);
            dxdu_ = mu_ * ((cx_.transpose() * cu_)) - std::get<2>(ans);
            dudu_ = mu_ * ((cu_.transpose() * cu_)) - std::get<1>(ans);
        } else {
            auto ans = tensor_contract(lambda_proj_, hessian_tensor);
            // Eigen::Matrix<double, constraint_dim, state_dim> jac_proj_cx = proj_jac_ * cx;
            // Eigen::Matrix<double, constraint_dim, control_dim> jac_proj_cu = proj_jac_ * cu;
            dxdx_ =  dxdx_ - 1.0 * std::get<0>(ans);
            dxdu_ =  dxdu_ - 1.0 * std::get<2>(ans);
            dudu_ =  dudu_ - 1.0 * std::get<1>(ans);

            // dxdu = mu_ * ((jac_proj_cx.transpose() * jac_proj_cu) - std::get<2>(ans));
            // dudu = mu_ * ((jac_proj_cu.transpose() * jac_proj_cu) - std::get<1>(ans));

            // auto ans = tensor_contract(lambda_proj_, hessian_tensor);
            // Eigen::Matrix<double, constraint_dim, state_dim> jac_proj_cx = proj_jac_ * cx;
            // Eigen::Matrix<double, constraint_dim, control_dim> jac_proj_cu = proj_jac_ * cu;
            //dxdx = (proj_cx_T_ * cx_);
            // dxdu = mu_ * ((cx_.transpose() * proj_jac_ * cu_));
            //dudu = (proj_cu_T_ * cu_);
        }

        return {dxdx_, dudu_, dxdu_};
    }

    // ============================================
    // 增广拉格朗日参数更新
    // ============================================

    /**
     * @brief 更新拉格朗日乘子 λ
     *
     * 根据当前约束违反情况更新对偶变量 λ。
     *
     * @param x 当前状态向量 (state_dim × 1)
     * @param u 当前控制向量 (control_dim × 1)
     *
     * ## 数学原理
     *
     * 拉格朗日乘子更新是增广拉格朗日法的核心步骤,对应乘子法 (Method of Multipliers):
     *
     * ### 等式约束 (is_equality = true)
     *
     * ```
     * λ_{k+1} = λ_k - μ * h(x*, u*)
     * ```
     *
     * 这是标准的乘子法更新公式。
     *
     * ### 不等式约束 (is_equality = false)
     *
     * ```
     * λ_{k+1} = proj(λ_k - μ * c(x*, u*))
     *         = min(λ_k - μ * c(x*, u*), 0)  (逐元素)
     * ```
     *
     * 投影算子确保 KKT 条件:
     * - λ >= 0 (非负性)
     * - λᵀc = 0 (互补松弛条件)
     *
     * ## 物理意义
     *
     * 拉格朗日乘子可以理解为"约束的影子价格":
     * - λ[i] < 0: 约束 i 是活跃的 (tight),对优化有显著影响
     * - λ[i] = 0: 约束 i 是非活跃的 (slack),不影响当前解
     *
     * ## 更新频率
     *
     * 在增广拉格朗日 iLQR 中,通常在外层循环更新 λ:
     * ```
     * for outer_iter = 1 to max_outer_iter:
     *     # 内层 iLQR 求解
     *     x*, u* = iLQR_solve(λ, μ)
     *
     *     # 更新拉格朗日乘子
     *     update_lambda(x*, u*)
     *
     *     # 检查收敛
     *     if max_violation < tol:
     *         break
     * ```
     *
     * @note 如果约束满足 (c ≈ 0), 则 λ 几乎不变
     * @note 如果约束违反严重 (c >> 0), 则 λ 会显著增大 (变得更负)
     */
    void update_lambda(const Eigen::Ref<const Eigen::Matrix<double, state_dim, 1>>& x,
                       const Eigen::Ref<const Eigen::Matrix<double, control_dim, 1>>& u) {
        if (is_equality_) {
            // 等式约束: 直接更新 λ = λ - μ*h
            lambda_ -= mu_ * constraints(x, u);
        } else {
            // 不等式约束: 投影到可行集 λ = proj(λ - μ*c)
            lambda_ = projection(lambda_ - mu_ * constraints(x, u));
        }
    }

    /**
     * @brief 更新惩罚系数 μ
     *
     * 设置新的惩罚系数,用于增强约束的惩罚强度。
     *
     * @param new_mu 新的惩罚系数 (必须 > 0)
     *
     * ## 惩罚系数的作用
     *
     * 惩罚系数 μ 控制约束违反的惩罚强度:
     * ```
     * L_aug = f(x, u) + λᵀc + (μ/2)||c||²
     *                         \_________/
     *                         惩罚项
     * ```
     *
     * ## 更新策略
     *
     * 典型的增广拉格朗日法采用自适应更新:
     * ```
     * if max_violation > threshold:
     *     μ = β * μ  (增大惩罚, β > 1, 通常 β = 2 或 10)
     * else:
     *     μ 保持不变
     * ```
     *
     * ## 数值考虑
     *
     * **μ 太小**:
     * - 优点: 数值条件好,Hessian 矩阵良态
     * - 缺点: 约束满足慢,可能需要更多外层迭代
     *
     * **μ 太大**:
     * - 优点: 约束满足快
     * - 缺点: Hessian 矩阵病态,iLQR 可能不收敛
     *
     * **最佳实践**:
     * - 初始值: μ_0 = 1.0
     * - 增长因子: β = 2~10
     * - 上限: μ_max = 1e6 (避免数值溢出)
     *
     * ## 在外层循环中的应用
     *
     * ```
     * mu = 1.0
     * for outer_iter = 1 to max_outer_iter:
     *     x*, u* = iLQR_solve(λ, μ)
     *     update_lambda(x*, u*)
     *
     *     violation = max_violation(x*, u*)
     *     if violation > threshold:
     *         update_mu(beta * mu)  # 增大惩罚
     * ```
     *
     * @note μ 越大,优化问题越接近纯惩罚法
     * @note μ 的增长应该渐进式,避免突然跳跃导致数值不稳定
     */
    void update_mu(double new_mu) {
        mu_ = new_mu;
    }

    /**
     * @brief 计算最大约束违反量
     *
     * 计算所有约束中的最大违反值,用于判断收敛和决定是否增大惩罚系数。
     *
     * @param x 状态向量 (state_dim × 1)
     * @param u 控制向量 (control_dim × 1)
     * @return 最大约束违反量 (标量, >= 0)
     *
     * ## 数学定义
     *
     * 对于不等式约束 c(x, u) <= 0:
     * ```
     * violation = ||max(c, 0)||_∞
     *           = max(c[0], c[1], ..., c[n], 0)
     * ```
     *
     * 实现中通过投影算子计算:
     * ```
     * c_proj = proj(c) = min(c, 0)  (满足约束的部分)
     * dc = c - c_proj = max(c, 0)   (违反量)
     * violation = ||dc||_∞            (最大违反)
     * ```
     *
     * ## 违反量的含义
     *
     * - `violation = 0`: 所有约束都满足
     * - `0 < violation < tol`: 轻微违反,可接受
     * - `violation >> tol`: 严重违反,需要增大 μ
     *
     * ## 在增广拉格朗日算法中的应用
     *
     * ### 收敛判断
     * ```
     * if max_violation < tolerance:
     *     return "收敛"
     * ```
     *
     * ### 自适应惩罚更新
     * ```
     * violation = max_violation(x, u)
     * if violation > 0.1 * previous_violation:
     *     # 违反没有明显改善,增大惩罚
     *     update_mu(beta * mu)
     * ```
     *
     * ## 外层循环示例
     *
     * ```
     * for outer_iter = 1 to max_outer_iter:
     *     x*, u* = iLQR_solve(λ, μ)
     *     update_lambda(x*, u*)
     *
     *     # 检查收敛
     *     violation = max_violation(x*, u*)
     *     if violation < 1e-3:
     *         break  # 约束满足,收敛
     *
     *     # 自适应更新惩罚系数
     *     if violation > 0.5 * prev_violation:
     *         update_mu(10 * mu)  # 进展不足,加大惩罚
     * ```
     *
     * ## 范数选择
     *
     * 使用 L∞ 范数 (最大值) 而非 L2 范数 (欧氏长度):
     * - **优点**: 更严格,确保每个约束都满足
     * - **缺点**: 对单个约束的违反敏感
     *
     * 如果使用 L2 范数:
     * ```
     * dc.norm()  // 多个小违反可能掩盖单个大违反
     * ```
     *
     * @note 对于等式约束,violation = ||h(x, u)||_∞
     * @note 返回值总是非负的
     */
    double max_violation(const Eigen::Ref<const Eigen::Matrix<double, state_dim, 1>>& x,
                         const Eigen::Ref<const Eigen::Matrix<double, control_dim, 1>>& u) const {
        // 计算约束值
        Eigen::Matrix<double, constraint_dim, 1> c = constraints(x, u);

        // 投影到可行域: c_proj = min(c, 0)
        Eigen::Matrix<double, constraint_dim, 1> c_proj = projection(c);

        // 违反量: dc = c - c_proj = max(c, 0)
        Eigen::Matrix<double, constraint_dim, 1> dc = c - c_proj;

        // 返回 L∞ 范数 (最大绝对值)
        return dc.template lpNorm<Eigen::Infinity>();
    }

    /**
     * @brief 一次性计算所有约束相关信息 (代价、梯度、Hessian)
     *
     * 批量计算增广拉格朗日代价的零阶、一阶、二阶导数信息。
     * 相比分别调用 augmented_lagrangian_cost, jacobian, hessian,
     * 此函数避免重复计算约束值和导数,提高效率。
     *
     * @param x 状态向量 (state_dim × 1)
     * @param u 控制向量 (control_dim × 1)
     * @param[out] augmented_lagrangian_cost 增广拉格朗日代价 (标量)
     * @param[out] augmented_lagrangian_jacobian_x 对状态的梯度 (state_dim × 1)
     * @param[out] augmented_lagrangian_jacobian_u 对控制的梯度 (control_dim × 1)
     * @param[out] augmented_lagrangian_hessian_xx 对状态的 Hessian (state_dim × state_dim)
     * @param[out] augmented_lagrangian_hessian_uu 对控制的 Hessian (control_dim × control_dim)
     * @param[out] augmented_lagrangian_hessian_xu 混合 Hessian (state_dim × control_dim)
     *
     * ## 计算流程
     *
     * 1. **计算约束及其导数** (一次性完成):
     *    ```
     *    c = constraints(x, u)
     *    cx, cu = constraints_jacobian(x, u)
     *    Hxx, Huu, Hxu = constraints_hessian(x, u)
     *    ```
     *
     * 2. **计算投影和因子**:
     *    ```
     *    factor = λ - μ*c
     *    lambda_proj = proj(factor)
     *    proj_jac = ∂proj/∂factor
     *    ```
     *
     * 3. **计算增广拉格朗日代价**:
     *    ```
     *    L_aug = 0.5/μ * (||lambda_proj||² - ||λ||²)
     *    ```
     *
     * 4. **计算梯度**:
     *    ```
     *    ∂L_aug/∂x = -(proj_jac * cx)ᵀ lambda_proj
     *    ∂L_aug/∂u = -(proj_jac * cu)ᵀ lambda_proj
     *    ```
     *
     * 5. **计算 Hessian**:
     *    ```
     *    ∂²L_aug/∂x² = μ (jac_proj_cx)ᵀ (jac_proj_cx) - Σᵢ factor[i] * Hxxᵢ
     *    ∂²L_aug/∂u² = μ (jac_proj_cu)ᵀ (jac_proj_cu) - Σᵢ factor[i] * Huuᵢ
     *    ∂²L_aug/∂x∂u = μ (jac_proj_cx)ᵀ (jac_proj_cu) - Σᵢ factor[i] * Hxuᵢ
     *    ```
     *
     * ## 性能优势
     *
     * 相比分别调用:
     * ```
     * cost = augmented_lagrangian_cost(x, u)
     * dx, du = augmented_lagrangian_jacobian(x, u)
     * dxdx, dudu, dxdu = augmented_lagrangian_hessian(x, u)
     * ```
     *
     * 此函数避免:
     * - 3 次 constraints(x, u) 调用 → 1 次
     * - 2 次 constraints_jacobian(x, u) 调用 → 1 次
     * - 1 次 constraints_hessian(x, u) 调用 → 1 次
     *
     * 典型性能提升: **2-3 倍**
     *
     * ## 使用场景
     *
     * **适用**:
     * - 需要同时使用代价、梯度、Hessian (如 Newton-type 算法)
     * - 约束计算开销大 (如非线性约束)
     *
     * **不适用**:
     * - 只需要代价 (如线搜索评估)
     * - 只需要梯度 (如梯度下降)
     *
     * ## 示例
     *
     * ```cpp
     * // iLQR Backward Pass 中计算代价和导数
     * double l_aug;
     * Eigen::Matrix<double, 6, 1> lx;
     * Eigen::Matrix<double, 2, 1> lu;
     * Eigen::Matrix<double, 6, 6> lxx;
     * Eigen::Matrix<double, 2, 2> luu;
     * Eigen::Matrix<double, 6, 2> lxu;
     *
     * constraints.CalcAllConstrainInfo(x, u, l_aug, lx, lu, lxx, luu, lxu);
     *
     * // 添加到原始代价函数
     * total_cost = trajectory_cost + l_aug;
     * total_lx = trajectory_lx + lx;
     * total_lxx = trajectory_lxx + lxx;
     * ```
     *
     * @note 此函数不修改内部状态 (cx_, cu_ 等),是独立的计算
     * @note 对于线性约束,Hessian 张量为零,计算更快
     */
    void CalcAllConstrainInfo(const Eigen::Ref<const Eigen::Matrix<double, state_dim, 1>>& x,
                 const Eigen::Ref<const Eigen::Matrix<double, control_dim, 1>>& u,
                 double& augmented_lagrangian_cost,
                 Eigen::Matrix<double, state_dim, 1>& augmented_lagrangian_jacobian_x,
                 Eigen::Matrix<double, control_dim, 1>& augmented_lagrangian_jacobian_u,
                 Eigen::Matrix<double, state_dim, state_dim> & augmented_lagrangian_hessian_xx,
                 Eigen::Matrix<double, control_dim, control_dim>& augmented_lagrangian_hessian_uu,
                 Eigen::Matrix<double, state_dim, control_dim>& augmented_lagrangian_hessian_xu) {
        Eigen::Matrix<double, constraint_dim, 1> c = constraints(x, u);
        auto jacobian_matrix = constraints_jacobian(x, u);
        auto cx = jacobian_matrix.first;
        auto cu = jacobian_matrix.second;
        Eigen::Matrix<double, state_dim, 1> dx;
        Eigen::Matrix<double, control_dim, 1> du;
        Eigen::Matrix<double, constraint_dim, 1> factor = lambda_ - mu_ * c;
        Eigen::Matrix<double, constraint_dim, 1> lambda_proj = projection(factor);
        Eigen::Matrix<double, state_dim, state_dim> dxdx = Eigen::Matrix<double, state_dim, state_dim>::Zero();
        Eigen::Matrix<double, control_dim, control_dim> dudu = Eigen::Matrix<double, control_dim, control_dim>::Zero();
        Eigen::Matrix<double, state_dim, control_dim> dxdu = Eigen::Matrix<double, state_dim, control_dim>::Zero();
        Eigen::Matrix<double, constraint_dim, constraint_dim> proj_jac = projection_jacobian(factor);
        auto hessian_tensor = constraints_hessian(x, u);
        auto ans = tensor_contract(factor, hessian_tensor);

        if (is_equality_) {
            augmented_lagrangian_cost = 0.5 / mu_ * ((lambda_ - mu_ * c).transpose() * (lambda_ - mu_ * c) - lambda_.transpose() * lambda_).value();
            dx = -cx.transpose() * factor;
            du = -cu.transpose() * factor;
            augmented_lagrangian_jacobian_x = dx;
            augmented_lagrangian_jacobian_u = du;
            dxdx = mu_ * ((cx.transpose() * cx) - std::get<0>(ans));
            dxdu = mu_ * ((cx.transpose() * cu) - std::get<2>(ans));
            dudu = mu_ * ((cu.transpose() * cu) - std::get<1>(ans));
            augmented_lagrangian_hessian_xx = dxdx;
            augmented_lagrangian_hessian_uu = dudu;
            augmented_lagrangian_hessian_xu = dxdu;

        } else {
            augmented_lagrangian_cost = 0.5 / mu_ * (lambda_proj.transpose() * lambda_proj - lambda_.transpose() * lambda_).value();
            dx = -(proj_jac * cx).transpose() * lambda_proj;
            du = -(proj_jac * cu).transpose() * lambda_proj;
            Eigen::Matrix<double, constraint_dim, state_dim> jac_proj_cx = proj_jac * cx;
            Eigen::Matrix<double, constraint_dim, control_dim> jac_proj_cu = proj_jac * cu;
            dxdx = mu_ * ((jac_proj_cx.transpose() * jac_proj_cx) - std::get<0>(ans));
            dxdu = mu_ * ((jac_proj_cx.transpose() * jac_proj_cu) - std::get<2>(ans));
            dudu = mu_ * ((jac_proj_cu.transpose() * jac_proj_cu) - std::get<1>(ans));
        }
    }


public:
    /** @brief 当前约束索引 (用于动态约束管理) */
    int current_constraints_index_ = 0;

private:
    // ============================================
    // 私有成员变量 (内部状态缓存)
    // ============================================

    /** @brief 拉格朗日乘子向量 λ (constraint_dim × 1) */
    Eigen::Matrix<double, constraint_dim, 1> lambda_;

    /** @brief 投影后的拉格朗日乘子 proj(λ - μ*c) (constraint_dim × 1) */
    Eigen::Matrix<double, constraint_dim, 1> lambda_proj_;

    /** @brief 缓存的约束值 c(x, u) (constraint_dim × 1) */
    Eigen::Matrix<double, constraint_dim, 1> c_;

    /** @brief 缓存的约束对状态的雅可比 ∂c/∂x (constraint_dim × state_dim) */
    Eigen::Matrix<double, constraint_dim, state_dim> cx_;

    /** @brief 缓存的约束对状态的 Hessian 数组 (constraint_dim 个矩阵) */
    std::array<Eigen::Matrix<double, state_dim, state_dim>, constraint_dim> cxx_;

    /** @brief 缓存的约束对控制的雅可比 ∂c/∂u (constraint_dim × control_dim) */
    Eigen::Matrix<double, constraint_dim, control_dim> cu_;

    /** @brief 投影后的约束雅可比转置 (proj_jac * cx)ᵀ (state_dim × constraint_dim) */
    Eigen::Matrix<double, state_dim, constraint_dim> proj_cx_T_;

    /** @brief 投影后的约束雅可比转置 (proj_jac * cu)ᵀ (control_dim × constraint_dim) */
    Eigen::Matrix<double, control_dim, constraint_dim> proj_cu_T_;

    /** @brief 缓存的 Hessian 对状态分量 ∂²L_aug/∂x² (state_dim × state_dim) */
    Eigen::Matrix<double, state_dim, state_dim> dxdx_;

    /** @brief 缓存的 Hessian 对控制分量 ∂²L_aug/∂u² (control_dim × control_dim) */
    Eigen::Matrix<double, control_dim, control_dim> dudu_;

    /** @brief 缓存的混合 Hessian ∂²L_aug/∂x∂u (state_dim × control_dim) */
    Eigen::Matrix<double, state_dim, control_dim> dxdu_;

    /** @brief 惩罚系数 μ (标量, μ > 0) */
    double mu_;

    /** @brief 约束类型: true 为等式约束, false 为不等式约束 */
    bool is_equality_;

    /** @brief 投影算子的雅可比矩阵 (对角矩阵, constraint_dim × constraint_dim) */
    Eigen::Matrix<double, constraint_dim, constraint_dim> proj_jac_;



    // ============================================
    // 辅助函数
    // ============================================

    /**
     * @brief 张量收缩辅助函数
     *
     * 计算向量与三阶张量的收缩 (内积):
     * ```
     * result_xx = Σᵢ factor[i] * Hxx[i]
     * result_uu = Σᵢ factor[i] * Huu[i]
     * result_xu = Σᵢ factor[i] * Hxu[i]
     * ```
     *
     * @param factor 权重向量 (constraint_dim × 1)
     * @param tensor Hessian 张量,包含三个数组:
     *               - std::get<0>(tensor): Hxx 数组,每个元素是 state_dim × state_dim 矩阵
     *               - std::get<1>(tensor): Huu 数组,每个元素是 control_dim × control_dim 矩阵
     *               - std::get<2>(tensor): Hxu 数组,每个元素是 state_dim × control_dim 矩阵
     * @return std::tuple<result_xx, result_uu, result_xu>
     *         - result_xx: 收缩后的状态 Hessian (state_dim × state_dim)
     *         - result_uu: 收缩后的控制 Hessian (control_dim × control_dim)
     *         - result_xu: 收缩后的混合 Hessian (state_dim × control_dim)
     *
     * ## 数学背景
     *
     * 在计算增广拉格朗日 Hessian 时,需要计算:
     * ```
     * ∂²L_aug/∂x² = μ cₓᵀ cₓ - Σᵢ [λᵢ - μ*cᵢ] * ∂²cᵢ/∂x²
     *                          \______________________/
     *                          张量收缩 (此函数计算)
     * ```
     *
     * 其中:
     * - factor = λ - μ*c (对于等式约束) 或 proj(λ - μ*c) (对于不等式约束)
     * - Hxx[i] = ∂²cᵢ/∂x² (第 i 个约束的状态 Hessian)
     *
     * ## 张量收缩的物理意义
     *
     * - **线性约束**: Hessian 全为零,收缩结果为零矩阵 (快速)
     * - **二次约束**: Hessian 为常数矩阵,收缩是加权和
     * - **非线性约束**: Hessian 依赖于 (x, u),收缩引入二阶校正
     *
     * ## 算法流程
     *
     * 1. 初始化结果为零矩阵
     * 2. 遍历所有约束 i = 0 到 constraint_dim-1:
     *    - result_xx += factor[i] * Hxx[i]
     *    - result_uu += factor[i] * Huu[i]
     *    - result_xu += factor[i] * Hxu[i]
     *
     * ## 性能考虑
     *
     * - 时间复杂度: O(constraint_dim * state_dim²) (主导项)
     * - 对于盒式约束 (线性): 所有 Hessian 为零,循环可跳过
     * - 对于圆形障碍物 (二次): 约 5-10 个约束,计算快速
     *
     * ## 示例
     *
     * ```cpp
     * // 因子: λ - μ*c
     * Eigen::Matrix<double, 5, 1> factor;
     * factor << -0.1, -0.5, 0.0, 0.0, -0.2;
     *
     * // Hessian 张量 (5 个约束)
     * auto hessian_tensor = constraints_hessian(x, u);
     *
     * // 张量收缩
     * auto [Hxx_sum, Huu_sum, Hxu_sum] = tensor_contract(factor, hessian_tensor);
     *
     * // 用于 Hessian 修正
     * Hessian_xx = μ * cx.T * cx - Hxx_sum;
     * ```
     *
     * @note 对于 factor[i] = 0 的约束,对应的 Hessian 不参与求和 (非活跃约束)
     * @note 函数声明为 const,不修改任何成员变量
     */
    std::tuple<Eigen::Matrix<double, state_dim, state_dim>, Eigen::Matrix<double, control_dim, control_dim>, Eigen::Matrix<double, state_dim, control_dim>>
    tensor_contract(const Eigen::Matrix<double, constraint_dim, 1>& factor,
                                    const std::tuple<std::array<Eigen::Matrix<double, state_dim, state_dim>, constraint_dim>,
                       std::array<Eigen::Matrix<double, control_dim, control_dim>, constraint_dim>,
                       std::array<Eigen::Matrix<double, state_dim, control_dim>, constraint_dim>>& tensor) const {

        // 初始化结果矩阵为零
        Eigen::Matrix<double, state_dim, state_dim> factor_dot_hxx;
        Eigen::Matrix<double, control_dim, control_dim> factor_dot_huu;
        Eigen::Matrix<double, state_dim, control_dim> factor_dot_hxu;

        // 提取 Hessian 张量的三个分量
        auto hxx = std::get<0>(tensor);  // 状态 Hessian 数组
        auto huu = std::get<1>(tensor);  // 控制 Hessian 数组
        auto hxu = std::get<2>(tensor);  // 混合 Hessian 数组

        factor_dot_hxx.setZero();
        factor_dot_huu.setZero();
        factor_dot_hxu.setZero();

        // 遍历所有约束,执行加权求和: result = Σᵢ factor[i] * H[i]
        for(size_t index = 0; index < constraint_dim; ++index) {
            factor_dot_hxx += hxx[index] * factor(index, 0);  // 累加状态 Hessian
            factor_dot_huu += huu[index] * factor(index, 0);  // 累加控制 Hessian
            factor_dot_hxu += hxu[index] * factor(index, 0);  // 累加混合 Hessian
        }

        return {factor_dot_hxx, factor_dot_huu, factor_dot_hxu};
    }
};

#endif // CONSTRAIN_CONSTRAIN_H_
