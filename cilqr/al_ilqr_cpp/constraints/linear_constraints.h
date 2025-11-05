#ifndef CONSTRAINTS_LINEAR_CONSTRAINTS_H_
#define CONSTRAINTS_LINEAR_CONSTRAINTS_H_

#include "constraints.h"

/**
 * @file linear_constraints.h
 * @brief 线性约束类 - 用于表示轨迹优化中的线性不等式/等式约束
 *
 * LinearConstraints 类实现了标准的线性约束形式,广泛应用于轨迹优化、运动规划等领域。
 * 线性约束是约束优化问题中最基本的约束类型,易于处理且计算效率高。
 */

/**
 * @brief 线性约束类 - 实现标准线性约束
 *
 * LinearConstraints 类继承自 Constraints 基类,用于表示线性不等式约束或等式约束。
 *
 * ## 约束形式
 *
 * 标准形式 (不等式约束):
 * ```
 * A*x + B*u + C ≤ 0
 * ```
 *
 * 等式约束形式:
 * ```
 * A*x + B*u + C = 0
 * ```
 *
 * 其中:
 * - x: state_dim 维状态向量
 * - u: control_dim 维控制向量
 * - A: constraint_dim × state_dim 矩阵 (状态系数矩阵)
 * - B: constraint_dim × control_dim 矩阵 (控制系数矩阵)
 * - C: constraint_dim × 1 向量 (常数项向量)
 *
 * ## 线性约束的特点
 *
 * 1. **雅可比矩阵为常数**: ∂c/∂x = A, ∂c/∂u = B (不依赖于 x, u 的值)
 * 2. **海森矩阵为零**: ∂²c/∂x² = 0, ∂²c/∂u² = 0, ∂²c/∂x∂u = 0
 * 3. **计算高效**: 约束评估和导数计算都是简单的矩阵乘法
 * 4. **凸性**: 线性约束定义的可行域是凸集
 *
 * ## 典型应用
 *
 * 1. **盒式约束 (Box Constraints)**: 状态和控制的上下界
 *    ```
 *    x_min ≤ x ≤ x_max  →  转换为线性约束
 *    u_min ≤ u ≤ u_max
 *    ```
 *
 * 2. **线性路径约束**: 例如车道保持、安全走廊
 *    ```
 *    a·x + b·y ≤ c  (表示半平面约束)
 *    ```
 *
 * 3. **控制速率限制**: 限制控制输入的变化率
 *    ```
 *    u_k - u_{k-1} ≤ Δu_max
 *    ```
 *
 * ## 使用示例
 *
 * ```cpp
 * // 示例: 2维状态, 1维控制, 4个约束
 * // 约束 1-2: x[0] ≤ 10, x[0] ≥ -5  (状态上下界)
 * // 约束 3-4: u[0] ≤ 1,  u[0] ≥ -1  (控制上下界)
 *
 * Eigen::Matrix<double, 4, 2> A;
 * A << 1,  0,    // 约束 1: x[0] - 10 ≤ 0
 *     -1,  0,    // 约束 2: -x[0] - 5 ≤ 0
 *      0,  0,    // 约束 3: 不涉及状态
 *      0,  0;    // 约束 4: 不涉及状态
 *
 * Eigen::Matrix<double, 4, 1> B;
 * B << 0,        // 约束 1: 不涉及控制
 *      0,        // 约束 2: 不涉及控制
 *      1,        // 约束 3: u[0] - 1 ≤ 0
 *     -1;        // 约束 4: -u[0] - 1 ≤ 0
 *
 * Eigen::Matrix<double, 4, 1> C;
 * C << -10,      // 约束 1 常数项
 *       -5,      // 约束 2 常数项
 *       -1,      // 约束 3 常数项
 *       -1;      // 约束 4 常数项
 *
 * LinearConstraints<2, 1, 4> constraints(A, B, C);
 *
 * // 评估约束
 * Eigen::Vector2d x(5.0, 0.0);
 * Eigen::Vector1d u(0.5);
 * auto c_val = constraints.constraints(x, u);
 * // c_val = [5-10, -5-5, 0.5-1, -0.5-1] = [-5, -10, -0.5, -1.5]
 * // 所有值 ≤ 0, 约束满足!
 * ```
 *
 * @tparam state_dim 状态维度
 * @tparam control_dim 控制维度
 * @tparam constraint_dim 约束个数 (约束维度)
 *
 * @see BoxConstraints 盒式约束的具体实现
 * @see Constraints 约束基类
 */
template <int state_dim, int control_dim, int constraint_dim>
class LinearConstraints : public Constraints<state_dim, control_dim, constraint_dim> {
public:
    // Eigen 内存对齐宏,确保在使用 SIMD 指令时正确对齐内存
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /**
     * @brief 构造线性约束对象
     *
     * 根据给定的系数矩阵 A, B, C 创建线性约束。
     *
     * @param A 状态系数矩阵 (constraint_dim × state_dim)
     *          定义了约束中状态 x 的线性系数
     * @param B 控制系数矩阵 (constraint_dim × control_dim)
     *          定义了约束中控制 u 的线性系数
     * @param C 常数项向量 (constraint_dim × 1)
     *          定义了约束中的常数偏置
     * @param is_equality 是否为等式约束 (默认 false 表示不等式约束 ≤ 0)
     *                    - false: 不等式约束 A*x + B*u + C ≤ 0
     *                    - true:  等式约束   A*x + B*u + C = 0
     *
     * @note 等式约束在增广拉格朗日方法中会被特殊处理:
     *       - 不等式约束: 只惩罚违反的约束 (max(0, c))
     *       - 等式约束: 总是惩罚偏离 (|c|)
     *
     * @example
     * ```cpp
     * // 创建不等式约束: x[0] + x[1] ≤ 10
     * Eigen::Matrix<double, 1, 2> A;
     * A << 1, 1;
     * Eigen::Matrix<double, 1, 1> B;
     * B << 0;
     * Eigen::Matrix<double, 1, 1> C;
     * C << -10;
     * LinearConstraints<2, 1, 1> constraint(A, B, C, false);
     * ```
     */
    LinearConstraints(const Eigen::Matrix<double, constraint_dim, state_dim>& A,
                      const Eigen::Matrix<double, constraint_dim, control_dim>& B,
                      const Eigen::Matrix<double, constraint_dim, 1>& C,
                      bool is_equality = false)
        : Constraints<state_dim, control_dim, constraint_dim>(is_equality), A_(A), B_(B), C_(C) {}
    /**
     * @brief 评估约束函数值
     *
     * 计算给定状态 x 和控制 u 下的约束函数值。
     *
     * 计算公式:
     * ```
     * c(x, u) = A*x + B*u + C
     * ```
     *
     * @param x 状态向量 (state_dim × 1)
     * @param u 控制向量 (control_dim × 1)
     * @return 约束函数值向量 (constraint_dim × 1)
     *         - 对于不等式约束: 返回值 ≤ 0 表示约束满足, > 0 表示约束违反
     *         - 对于等式约束: 返回值 = 0 表示约束满足, ≠ 0 表示约束违反
     *
     * @note 此方法被标记为 const override,表示:
     *       - const: 不修改对象状态
     *       - override: 覆盖基类的虚函数
     *
     * @example
     * ```cpp
     * // 假设约束为: x[0] + x[1] ≤ 10  (即 x[0] + x[1] - 10 ≤ 0)
     * Eigen::Vector2d x(3.0, 4.0);
     * Eigen::Vector1d u(0.0);
     * auto c = constraints.constraints(x, u);
     * // c = 3 + 4 - 10 = -3 < 0, 约束满足
     * ```
     */
    Eigen::Matrix<double, constraint_dim, 1> constraints(const Eigen::Ref<const Eigen::Matrix<double, state_dim, 1>>& x,
                                                       const Eigen::Ref<const Eigen::Matrix<double, control_dim, 1>>& u) const override {
        Eigen::Matrix<double, constraint_dim, 1> Ax = A_ * x;
        Eigen::Matrix<double, constraint_dim, 1> Bu = B_ * u;
        return Ax + Bu + C_;
    }

    /**
     * @brief 并行评估约束函数值 (向量化版本)
     *
     * 同时评估 PARALLEL_NUM 组状态和控制下的约束函数值,用于并行线搜索等场景。
     *
     * 计算公式:
     * ```
     * c_i(x_i, u_i) = A*x_i + B*u_i + C,  i = 1, ..., PARALLEL_NUM
     * ```
     *
     * @param x 状态矩阵 (state_dim × PARALLEL_NUM)
     *          每一列是一个状态向量
     * @param u 控制矩阵 (control_dim × PARALLEL_NUM)
     *          每一列是一个控制向量
     * @return 约束函数值矩阵 (constraint_dim × PARALLEL_NUM)
     *         每一列对应一组 (x, u) 的约束评估结果
     *
     * @note PARALLEL_NUM 是全局定义的并行度,通常在编译时指定
     *       例如 PARALLEL_NUM = 8 表示同时评估 8 组轨迹候选
     *
     * @note 实现细节:
     *       - C_.replicate(1, PARALLEL_NUM) 将 C 向量复制 PARALLEL_NUM 次
     *       - 矩阵乘法 A*x 和 B*u 会自动并行处理所有列
     *
     * @example
     * ```cpp
     * // 并行评估 8 组不同的状态和控制
     * Eigen::Matrix<double, 2, 8> x_parallel;  // 8 个状态向量
     * Eigen::Matrix<double, 1, 8> u_parallel;  // 8 个控制向量
     * // ... 填充 x_parallel 和 u_parallel ...
     *
     * auto c_parallel = constraints.parallel_constraints(x_parallel, u_parallel);
     * // c_parallel 是 (constraint_dim × 8) 矩阵,每列是一组约束评估
     * ```
     */
    Eigen::Matrix<double, constraint_dim, PARALLEL_NUM> parallel_constraints(const Eigen::Ref<const Eigen::Matrix<double, state_dim, PARALLEL_NUM>>& x,
                                                 const Eigen::Ref<const Eigen::Matrix<double, control_dim, PARALLEL_NUM>>& u) const {
        Eigen::Matrix<double, constraint_dim, PARALLEL_NUM> C_parallel = C_.replicate(1, PARALLEL_NUM);
        Eigen::Matrix<double, constraint_dim, PARALLEL_NUM> Ax = A_ * x;
        Eigen::Matrix<double, constraint_dim, PARALLEL_NUM> Bu = B_ * u;
        return Ax + Bu + C_parallel;
    }

    /**
     * @brief 计算约束函数的雅可比矩阵
     *
     * 计算约束函数对状态和控制的一阶偏导数。
     *
     * 对于线性约束 c(x, u) = A*x + B*u + C, 其雅可比矩阵为常数:
     * ```
     * ∂c/∂x = A  (constraint_dim × state_dim 矩阵)
     * ∂c/∂u = B  (constraint_dim × control_dim 矩阵)
     * ```
     *
     * 注意: 线性约束的雅可比矩阵不依赖于 x 和 u 的具体值!
     *
     * @param x 状态向量 (state_dim × 1) [未使用,仅为接口一致性]
     * @param u 控制向量 (control_dim × 1) [未使用,仅为接口一致性]
     * @return std::pair<∂c/∂x, ∂c/∂u>
     *         - first:  ∂c/∂x = A (constraint_dim × state_dim)
     *         - second: ∂c/∂u = B (constraint_dim × control_dim)
     *
     * @note 在 iLQR 算法中,雅可比矩阵用于:
     *       1. Backward Pass: 计算 Q 函数的梯度
     *       2. 增广拉格朗日项的梯度计算
     *
     * @note 对于线性约束,此方法计算非常高效 (O(1)),
     *       因为只需返回预存储的 A 和 B 矩阵
     *
     * @example
     * ```cpp
     * auto [A_jac, B_jac] = constraints.constraints_jacobian(x, u);
     * // A_jac == A_ (恒定不变)
     * // B_jac == B_ (恒定不变)
     * ```
     */
    std::pair<Eigen::Matrix<double, constraint_dim, state_dim>, Eigen::Matrix<double, constraint_dim, control_dim>>
    constraints_jacobian(const Eigen::Ref<const Eigen::Matrix<double, state_dim, 1>>& x,
                        const Eigen::Ref<const Eigen::Matrix<double, control_dim, 1>>& u) const override {
        return std::make_pair(A_, B_);
    }

    /**
     * @brief 计算约束函数的海森矩阵 (二阶导数)
     *
     * 计算约束函数对状态和控制的二阶偏导数。
     *
     * 对于线性约束 c(x, u) = A*x + B*u + C, 所有二阶导数恒为零:
     * ```
     * ∂²c_i/∂x²   = 0  (state_dim × state_dim 零矩阵)
     * ∂²c_i/∂u²   = 0  (control_dim × control_dim 零矩阵)
     * ∂²c_i/∂x∂u  = 0  (state_dim × control_dim 零矩阵)
     * ```
     * 其中 i = 1, ..., constraint_dim
     *
     * @param x 状态向量 (state_dim × 1) [未使用,仅为接口一致性]
     * @param u 控制向量 (control_dim × 1) [未使用,仅为接口一致性]
     * @return std::tuple<hxx, huu, hxu>
     *         - hxx: std::array of ∂²c_i/∂x² (constraint_dim 个零矩阵)
     *         - huu: std::array of ∂²c_i/∂u² (constraint_dim 个零矩阵)
     *         - hxu: std::array of ∂²c_i/∂x∂u (constraint_dim 个零矩阵)
     *
     * @note 在 iLQR 算法中,海森矩阵用于:
     *       1. Backward Pass: 计算 Q 函数的海森矩阵
     *       2. 增广拉格朗日项的二阶导数计算
     *
     * @note 对于线性约束,海森矩阵恒为零,这简化了 iLQR 的计算
     *       (与二次约束相比,二次约束的 ∂²c/∂x² 不为零)
     *
     * @note 返回值使用 std::array 而非 std::vector,
     *       因为 constraint_dim 在编译时已知,可以使用栈分配
     *
     * @example
     * ```cpp
     * auto [hxx, huu, hxu] = constraints.constraints_hessian(x, u);
     * // hxx[i] 是第 i 个约束的状态海森矩阵 (全零)
     * // huu[i] 是第 i 个约束的控制海森矩阵 (全零)
     * // hxu[i] 是第 i 个约束的混合海森矩阵 (全零)
     * ```
     */
    std::tuple<std::array<Eigen::Matrix<double, state_dim, state_dim>, constraint_dim>,
                       std::array<Eigen::Matrix<double, control_dim, control_dim>, constraint_dim>,
                       std::array<Eigen::Matrix<double, state_dim, control_dim>, constraint_dim>>
    constraints_hessian(const Eigen::Ref<const Eigen::Matrix<double, state_dim, 1>>& x,
                       const Eigen::Ref<const Eigen::Matrix<double, control_dim, 1>>& u) const override {

        // 创建零矩阵元素
        Eigen::Matrix<double, state_dim, state_dim> hxx_ele;
        Eigen::Matrix<double, control_dim, control_dim> huu_ele;
        Eigen::Matrix<double, state_dim, control_dim> hxu_ele;

        hxx_ele.setZero();
        huu_ele.setZero();
        hxu_ele.setZero();

        // 创建数组并填充为零矩阵
        std::array<Eigen::Matrix<double, state_dim, state_dim>, constraint_dim> hxx;
        std::array<Eigen::Matrix<double, control_dim, control_dim>, constraint_dim> huu;
        std::array<Eigen::Matrix<double, state_dim, control_dim>, constraint_dim> hxu;

        std::fill(hxx.begin(), hxx.end(), hxx_ele);
        std::fill(huu.begin(), huu.end(), huu_ele);
        std::fill(hxu.begin(), hxu.end(), hxu_ele);

        return std::make_tuple(hxx, huu, hxu);
    }

    /**
     * @brief 动态更新约束 (运行时添加新约束)
     *
     * 在优化过程中动态添加新的线性约束。如果约束已存在,则不重复添加。
     * 这个功能主要用于动态障碍物避障等场景。
     *
     * @param A_rows 新约束的 A 矩阵行向量 (1 × state_dim)
     *               定义新约束中状态 x 的系数
     * @param C_rows 新约束的 C 向量元素 (标量)
     *               定义新约束的常数项
     *
     * @note 此方法假设新约束不涉及控制 u (即 B 的对应行为零)
     * @note 约束存在性检查通过 C 向量值判断 (可能不够精确)
     * @note 会自动递增 current_constraints_index_ 来跟踪活动约束数量
     *
     * @warning 这个方法修改了约束对象的状态 (A_ 和 C_),
     *          因此不是线程安全的
     *
     * @warning 对于静态约束 (如盒式约束),通常不需要调用此方法,
     *          因为所有约束在构造时已经确定
     *
     * @example
     * ```cpp
     * // 动态添加约束: x[0] + 2*x[1] ≤ 5
     * Eigen::Matrix<double, 1, 2> new_A;
     * new_A << 1, 2;
     * double new_C = -5;  // 注意: 5 变为 -5
     * constraints.UpdateConstraints(new_A, new_C);
     * ```
     */
    void UpdateConstraints(const Eigen::Ref<const Eigen::Matrix<double, 1, state_dim>> A_rows, double C_rows) override {
        // 检查新约束是否已存在 (通过 C 向量值判断)
        bool exist = (C_.array() == C_rows).any();
        if (exist) {
            return;  // 约束已存在,无需添加
        }

        // 递增约束索引
        this->current_constraints_index_ = this->current_constraints_index_ + 1;
        // 更新 A 矩阵的对应行
        A_.row(this->current_constraints_index_) = A_rows;
        // 更新 C 向量的对应元素
        C_[this->current_constraints_index_] = C_rows;
    }

public:
    /**
     * @brief 状态系数矩阵 A (constraint_dim × state_dim)
     *
     * 定义约束中状态 x 的线性系数:
     * ```
     * c(x, u) = A*x + B*u + C
     * ```
     *
     * 每一行对应一个约束,每一列对应状态向量的一个维度。
     *
     * @example
     * ```
     * A = [1  0  0]  ← 约束 1: x[0] + ...
     *     [0  1  0]  ← 约束 2: x[1] + ...
     *     [1  1  0]  ← 约束 3: x[0] + x[1] + ...
     * ```
     */
    Eigen::Matrix<double, constraint_dim, state_dim> A_;

    /**
     * @brief 控制系数矩阵 B (constraint_dim × control_dim)
     *
     * 定义约束中控制 u 的线性系数:
     * ```
     * c(x, u) = A*x + B*u + C
     * ```
     *
     * 每一行对应一个约束,每一列对应控制向量的一个维度。
     *
     * @example
     * ```
     * B = [1  0]  ← 约束 1: ... + u[0]
     *     [0  1]  ← 约束 2: ... + u[1]
     *     [0  0]  ← 约束 3: 不涉及控制
     * ```
     */
    Eigen::Matrix<double, constraint_dim, control_dim> B_;

    /**
     * @brief 常数项向量 C (constraint_dim × 1)
     *
     * 定义约束中的常数偏置:
     * ```
     * c(x, u) = A*x + B*u + C
     * ```
     *
     * 每个元素对应一个约束的常数项。
     *
     * @note 对于上界约束 x ≤ max, C 的对应元素为 -max
     * @note 对于下界约束 x ≥ min, C 的对应元素为 +min
     *       (详见 box_constraints.h 的详细推导)
     *
     * @example
     * ```
     * C = [-10]  ← 约束 1: A*x + B*u - 10 ≤ 0
     *     [ -5]  ← 约束 2: A*x + B*u - 5 ≤ 0
     *     [  3]  ← 约束 3: A*x + B*u + 3 ≤ 0
     * ```
     */
    Eigen::Matrix<double, constraint_dim, 1> C_;
};
#endif // CONSTRAIN_LINEAR_CONSTRAINTS_H_
