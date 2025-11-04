#ifndef CONSTRAINTS_BOX_CONSTRAINTS_H_
#define CONSTRAINTS_BOX_CONSTRAINTS_H_

#include "linear_constraints.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>

/**
 * @brief 盒式约束类 - 实现状态和控制的上下界约束
 *
 * BoxConstraints 类继承自 LinearConstraints,用于表示状态和控制的盒式约束。
 * 盒式约束是最常见的约束类型,通过将上下界约束转换为线性不等式约束的形式:
 *
 * 约束形式: A*x + B*u + C <= 0
 *
 * 其中:
 * - x: state_dim 维状态向量
 * - u: control_dim 维控制向量
 * - 约束维度: 2*(state_dim + control_dim) (每个变量有上下界两个约束)
 *
 * 原始盒式约束:
 *   state_min <= x <= state_max
 *   control_min <= u <= control_max
 *
 * 转换为线性约束:
 *   x <= state_max  -->  x - state_max <= 0
 *   -x <= -state_min  -->  -x + state_min <= 0
 *   u <= control_max  -->  u - control_max <= 0
 *   -u <= -control_min  -->  -u + control_min <= 0
 *
 * @tparam state_dim 状态维度
 * @tparam control_dim 控制维度
 */
template <int state_dim, int control_dim>
class BoxConstraints : public LinearConstraints<state_dim, control_dim, 2 * (state_dim + control_dim)> {
public:
    // Eigen 内存对齐宏,确保在使用 SIMD 指令时正确对齐内存
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /**
     * @brief 构造盒式约束对象
     *
     * 根据给定的状态和控制的上下界,构造线性约束矩阵 A, B, C。
     *
     * @param state_min 状态的下界 (state_dim × 1 向量)
     * @param state_max 状态的上界 (state_dim × 1 向量)
     * @param control_min 控制的下界 (control_dim × 1 向量)
     * @param control_max 控制的上界 (control_dim × 1 向量)
     *
     * @note 约束矩阵的生成规则:
     *   - A 矩阵: [I; -I; 0; 0] 大小为 (2*state_dim + 2*control_dim) × state_dim
     *   - B 矩阵: [0; 0; I; -I] 大小为 (2*state_dim + 2*control_dim) × control_dim
     *   - C 矩阵: [-state_max; state_min; -control_max; control_min] 大小为 (2*state_dim + 2*control_dim) × 1
     *
     * @see generateA(), generateB(), generateC()
     */
    BoxConstraints(const Eigen::Matrix<double, state_dim, 1>& state_min,
                   const Eigen::Matrix<double, state_dim, 1>& state_max,
                   const Eigen::Matrix<double, control_dim, 1>& control_min,
                   const Eigen::Matrix<double, control_dim, 1>& control_max)
        : LinearConstraints<state_dim, control_dim, 2 * (state_dim + control_dim)>(generateA(state_min, state_max),
                                                                                   generateB(control_min, control_max),
                                                                                   generateC(state_min, state_max, control_min, control_max)) {}
    /**
     * @brief 动态更新约束 (当前实现为检查是否已存在)
     *
     * 此方法用于在运行时动态添加新的约束条件。如果要添加的约束已经存在(通过检查 C 向量),
     * 则不进行任何操作;否则将新约束添加到约束集合中。
     *
     * @param A_rows 新约束的 A 矩阵行向量 (1 × state_dim)
     * @param C_rows 新约束的 C 向量元素 (标量)
     *
     * @note 此方法覆盖了基类的虚函数
     * @note 检查约束是否存在的方式是比较 C 向量的值,这在某些情况下可能不够精确
     * @note 此方法会递增 current_constraints_index_,用于跟踪活动约束的数量
     *
     * @warning 对于盒式约束,通常不需要动态更新,因为所有约束在构造时已确定
     */
    void UpdateConstraints(const Eigen::Ref<const Eigen::Matrix<double, 1, state_dim>> A_rows, double C_rows) override {
        // 检查新约束是否已存在于 C_ 向量中
        bool exist = (this->C_.array() == C_rows).any();
        if (exist) {
            return;  // 约束已存在,无需添加
        }
        // 递增约束索引
        this->current_constraints_index_ = this->current_constraints_index_ + 1;
        // 添加新的 A 矩阵行
        this->A_.row(this->current_constraints_index_) = A_rows;
        // 添加新的 C 向量元素
        this->C_[this->current_constraints_index_] =  C_rows;
    }

private:
    /**
     * @brief 生成约束中的 A 矩阵 (状态相关部分)
     *
     * A 矩阵定义了约束中状态 x 的系数。对于盒式约束,A 矩阵的结构为:
     *
     * A = [ I_{state_dim}      ]  <- 对应 x <= state_max
     *     [-I_{state_dim}      ]  <- 对应 -x <= -state_min
     *     [ 0_{control_dim×state_dim} ]  <- 控制约束对状态无影响
     *     [ 0_{control_dim×state_dim} ]
     *
     * 其中 I 为单位矩阵,0 为零矩阵
     *
     * @param state_min 状态下界 (未使用,保留参数用于接口一致性)
     * @param state_max 状态上界 (未使用,保留参数用于接口一致性)
     * @return A 矩阵,大小为 (2*state_dim + 2*control_dim) × state_dim
     *
     * @note state_min 和 state_max 参数在此方法中未使用,实际约束边界体现在 C 矩阵中
     */
    static Eigen::Matrix<double, 2 * state_dim + 2 * control_dim, state_dim> generateA(const Eigen::Matrix<double, state_dim, 1>& state_min,
                                                                                       const Eigen::Matrix<double, state_dim, 1>& state_max) {
        // 构造状态约束部分: [I; -I]
        Eigen::Matrix<double, 2 * state_dim, state_dim> A_state;
        A_state << Eigen::Matrix<double, state_dim, state_dim>::Identity(),     // 上界约束
                   -Eigen::Matrix<double, state_dim, state_dim>::Identity();    // 下界约束

        // 组装完整的 A 矩阵: [A_state; 0]
        Eigen::Matrix<double, 2 * state_dim + 2 * control_dim, state_dim> A;
        A << A_state,
             Eigen::Matrix<double, 2 * control_dim, state_dim>::Zero();  // 控制约束对状态无贡献
        return A;
    }

    /**
     * @brief 生成约束中的 B 矩阵 (控制相关部分)
     *
     * B 矩阵定义了约束中控制 u 的系数。对于盒式约束,B 矩阵的结构为:
     *
     * B = [ 0_{state_dim×control_dim} ]  <- 状态约束对控制无影响
     *     [ 0_{state_dim×control_dim} ]
     *     [ I_{control_dim}      ]  <- 对应 u <= control_max
     *     [-I_{control_dim}      ]  <- 对应 -u <= -control_min
     *
     * 其中 I 为单位矩阵,0 为零矩阵
     *
     * @param control_min 控制下界 (未使用,保留参数用于接口一致性)
     * @param control_max 控制上界 (未使用,保留参数用于接口一致性)
     * @return B 矩阵,大小为 (2*state_dim + 2*control_dim) × control_dim
     *
     * @note control_min 和 control_max 参数在此方法中未使用,实际约束边界体现在 C 矩阵中
     */
    static Eigen::Matrix<double, 2 * state_dim + 2 * control_dim, control_dim> generateB(const Eigen::Matrix<double, control_dim, 1>& control_min,
                                                                                       const Eigen::Matrix<double, control_dim, 1>& control_max) {
        // 构造控制约束部分: [I; -I]
        Eigen::Matrix<double, 2 * control_dim, control_dim> B_control;
        B_control << Eigen::Matrix<double, control_dim, control_dim>::Identity(),   // 上界约束
                     -Eigen::Matrix<double, control_dim, control_dim>::Identity();  // 下界约束

        // 组装完整的 B 矩阵: [0; B_control]
        Eigen::Matrix<double, 2 * state_dim + 2 * control_dim, control_dim> B;
        B << Eigen::Matrix<double, 2 * state_dim, control_dim>::Zero(),  // 状态约束对控制无贡献
             B_control;
        return B;
    }

    /**
     * @brief 生成约束中的 C 向量 (常数项部分)
     *
     * C 向量定义了约束中的常数项,它包含了实际的上下界值。对于盒式约束,C 向量的结构为:
     *
     * C = [ -state_max    ]  <- 对应约束 x - state_max <= 0
     *     [  state_min    ]  <- 对应约束 -x + state_min <= 0 (即 x >= state_min)
     *     [ -control_max  ]  <- 对应约束 u - control_max <= 0
     *     [  control_min  ]  <- 对应约束 -u + control_min <= 0 (即 u >= control_min)
     *
     * 完整约束形式: A*x + B*u + C <= 0
     *
     * 示例:
     *   原始约束: x1 <= 10
     *   线性形式: x1 - 10 <= 0
     *   矩阵形式: A[0,:]*x + C[0] <= 0,其中 A[0,:] = [1, 0, ...], C[0] = -10
     *
     * @param state_min 状态的下界 (state_dim × 1 向量)
     * @param state_max 状态的上界 (state_dim × 1 向量)
     * @param control_min 控制的下界 (control_dim × 1 向量)
     * @param control_max 控制的上界 (control_dim × 1 向量)
     * @return C 向量,大小为 (2*state_dim + 2*control_dim) × 1
     *
     * @note 注意符号:上界对应负号(-state_max),下界对应正号(state_min)
     */
    static Eigen::Matrix<double, 2 * state_dim + 2 * control_dim, 1> generateC(const Eigen::Matrix<double, state_dim, 1>& state_min,
                                                                              const Eigen::Matrix<double, state_dim, 1>& state_max,
                                                                              const Eigen::Matrix<double, control_dim, 1>& control_min,
                                                                              const Eigen::Matrix<double, control_dim, 1>& control_max) {
        // 构造状态约束的常数项: [-state_max; state_min]
        Eigen::Matrix<double, 2 * state_dim, 1> C_state;
        C_state << -state_max,  // 上界约束的常数项
                    state_min;  // 下界约束的常数项

        // 构造控制约束的常数项: [-control_max; control_min]
        Eigen::Matrix<double, 2 * control_dim, 1> C_control;
        C_control << -control_max,  // 上界约束的常数项
                     control_min;   // 下界约束的常数项

        // 组装完整的 C 向量: [C_state; C_control]
        Eigen::Matrix<double, 2 * state_dim + 2 * control_dim, 1> C;
        C << C_state,
             C_control;
        return C;
    }


};  // class BoxConstraints

#endif // CONSTRAINTS_BOX_CONSTRAINTS_H_
