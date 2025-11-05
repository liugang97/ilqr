/**
 * @file test_new_al_ilqr.cc
 * @brief NewALILQR 求解器的测试程序
 *
 * 本文件测试完整动力学自行车模型 (6维状态) 的 iLQR 优化求解器，包括：
 * 1. 使用盒式约束 (BoxConstraints) 的轨迹优化
 * 2. 使用二次约束 (QuadraticConstraints) 处理圆形障碍物避障的轨迹优化
 */

#include "constraints/box_constraints.h"
#include "constraints/quadratic_constraints.h"
#include "new_al_ilqr.h"
#include <memory>
#include <vector>
#include <array>

/**
 * @brief 生成 S 形参考轨迹（完整状态）
 *
 * 生成一条平滑的 S 形参考轨迹，轨迹形状为正弦曲线，用于测试车辆跟踪性能。
 *
 * @param v 期望的前向速度 (m/s)
 * @param dt 时间步长 (s)
 * @param num_points 轨迹点数量
 * @return std::vector<Eigen::VectorXd> 目标状态序列，每个状态为 6 维向量 [x, y, θ, δ, v, a]
 *
 * 状态量说明：
 * - x, y: 位置坐标 (m)
 * - θ (theta): 车辆航向角 (rad)
 * - δ (delta): 前轮转角 (rad)
 * - v: 纵向速度 (m/s)
 * - a: 纵向加速度 (m/s²)
 */
std::vector<Eigen::VectorXd> generateSShapeGoalFull(double v, double dt, int num_points) {
    std::vector<Eigen::VectorXd> goals;
    for (int i = 0; i <= num_points; ++i) {
        double t = i * dt;  // 当前时间

        // 轨迹参数方程: x(t) = v*t, y(t) = 50*sin(0.1*t)
        // 这将生成一条在 x 方向匀速前进、在 y 方向正弦摆动的 S 形曲线
        double x = v * t;                      // x 坐标：匀速前进
        double y = 50 * std::sin(0.1 * t);     // y 坐标：正弦摆动，振幅 50m，频率 0.1 rad/s

        // 计算航向角 θ = atan2(dy/dt, dx/dt)
        // dx/dt = v, dy/dt = 50 * 0.1 * cos(0.1*t)
        double theta = std::atan2(50 * 0.1 * std::cos(0.1 * t), v);

        // 计算曲率 κ = (dx*d²y - dy*d²x) / (dx² + dy²)^1.5
        double dx = v;                                  // x 的一阶导数
        double dy = 50 * 0.1 * std::cos(0.1 * t);      // y 的一阶导数
        double ddx = 0;                                 // x 的二阶导数（匀速，加速度为 0）
        double ddy = -50 * 0.1 * 0.1 * std::sin(0.1 * t);  // y 的二阶导数
        double curvature = (dx * ddy - dy * ddx) / std::pow(dx * dx + dy * dy, 1.5);

        // 根据自行车模型几何关系：δ = atan(L * κ)，这里假设 L = 1.0
        double delta = std::atan(curvature * 1.0);

        // 构造 6 维目标状态向量
        Eigen::VectorXd goal_state(6);
        goal_state << x, y, theta, delta, v, 0;  // (x, y, theta, delta, v_desire, a_desire)
        goals.push_back(goal_state);
    }
    return goals;
}

/**
 * @brief 生成圆形障碍物的二次约束方程系数p
 *
 * 将圆形障碍物转换为二次约束的标准形式：x^T*Q*x + A^T*x + C <= 0
 * 圆形障碍物方程: (x - centre_x)² + (y - centre_y)² >= r²
 * 转换为约束形式: -(x - centre_x)² - (y - centre_y)² + r² <= 0
 * 展开后: -x² - y² + 2*centre_x*x + 2*centre_y*y + (r² - centre_x² - centre_y²) <= 0
 *
 * @param centre_x 圆心 x 坐标 (m)
 * @param centre_y 圆心 y 坐标 (m)
 * @param r 圆形障碍物半径 (m)
 * @param x_dims 状态向量维度（通常为 6）
 * @return std::tuple<Q, A, C> 二次约束系数矩阵
 *   - Q: 二次项系数矩阵 (x_dims × x_dims)，只有 Q(0,0) = Q(1,1) = -1（对应 x² 和 y²）
 *   - A: 一次项系数向量 (1 × x_dims)，A(0) = 2*centre_x, A(1) = 2*centre_y
 *   - C: 常数项 (1 × 1)，C = r² - centre_x² - centre_y²
 */
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> GenerateCycleEquations(double centre_x, double centre_y, double r, int x_dims) {
    Eigen::MatrixXd Q(x_dims, x_dims);
    Eigen::MatrixXd A(1, x_dims);
    Eigen::MatrixXd C(1, 1);

    // 常数项：r² - centre_x² - centre_y²
    C << r * r - centre_x * centre_x - centre_y * centre_y;

    // 初始化为零矩阵
    Q.setZero();
    A.setZero();

    // 二次项系数：-x² 和 -y²
    Q(0, 0) = -1.0;  // x² 的系数
    Q(1, 1) = -1.0;  // y² 的系数

    // 一次项系数：2*centre_x*x 和 2*centre_y*y
    A(0, 0) = 2 * centre_x;  // x 的系数
    A(0, 1) = 2 * centre_y;  // y 的系数

    return {Q, A, C};
}

/**
 * @brief 主函数 - 测试 NewALILQR 求解器
 *
 * 测试流程：
 * 1. 测试一：使用盒式约束进行轨迹优化（模拟简单的状态和控制界限）
 * 2. 测试二：使用二次约束进行轨迹优化（模拟圆形障碍物避障）
 */
int main() {
    // ========== 第一部分：参数设置 ==========

    // 基本参数
    double v = 10;          // 期望速度 (m/s)
    double dt = 0.1;        // 时间步长 (s)
    double L = 3.0;         // 车辆轴距 (m)
    int num_points = 50;    // 轨迹离散点数量

    // 生成 S 形参考轨迹
    std::vector<Eigen::VectorXd> goal_list_fast = generateSShapeGoalFull(v, dt, num_points);

    // 状态代价权重矩阵 Q (6×6)
    // 对角线元素分别对应 [x, y, θ, δ, v, a] 的权重
    Eigen::MatrixXd Q_fast = Eigen::MatrixXd::Zero(6, 6);
    Q_fast.diagonal() << 1e-1, 1e-1, 1e-0, 1e-9, 1e-6, 1e-6;
    Q_fast *= 1e3;  // 整体缩放因子
    std::cout << "Q_fast diagonal: " <<  std::endl;
    std::cout << Q_fast << std::endl;
    // 最终权重: [100, 100, 1000, 1e-6, 1e-3, 1e-3]
    // 说明：位置 (x,y) 和航向角 (θ) 的跟踪权重较高，其他状态权重较低

    // 控制代价权重矩阵 R (2×2)
    // 对角线元素分别对应 [δ_rate, a_rate] 的权重
    Eigen::MatrixXd R_fast = Eigen::MatrixXd::Identity(2, 2) * 1e2;
    std::cout << "R_fast diagonal: " <<  std::endl;
    std::cout << R_fast << std::endl;
    // 权重: [100, 100]，惩罚过大的转向速率和加速度变化率


    // ========== 测试一：使用盒式约束的轨迹优化 ==========

    // 定义状态约束边界 [下界, 上界]
    std::array<Eigen::Matrix<double, 6, 1>, 2> state_bounds;
    // 状态下界: [x_min, y_min, θ_min, δ_min, v_min, a_min]
    state_bounds[0] << -1000, -1000, -2 * M_PI, -10, -100, -10;
    // 状态上界: [x_max, y_max, θ_max, δ_max, v_max, a_max]
    state_bounds[1] << 1000, 1000, 2 * M_PI, 10, 100, 10;

    // 定义控制约束边界 [下界, 上界]
    std::array<Eigen::Matrix<double, 2, 1>, 2> control_bounds;
    // 控制下界: [δ_rate_min, a_rate_min]
    control_bounds[0] << -0.2, -1;
    // 控制上界: [δ_rate_max, a_rate_max]
    control_bounds[1] << 0.2, 1;

    // 创建盒式约束对象
    BoxConstraints<6, 2> constraints_obj(state_bounds[0], state_bounds[1], control_bounds[0], control_bounds[1]);

    // 创建 iLQR 节点列表（每个时间步一个节点）
    std::vector<std::shared_ptr<NewILQRNode<6, 2>>> ilqr_node_list;
    ilqr_node_list.clear();

    // 为每个时间步创建自行车模型节点
    for (int i = 0; i <= num_points; ++i) {
        // 参数: (轴距, 时间步长, 正则化系数, 目标状态, Q矩阵, R矩阵, 约束对象)
        ilqr_node_list.push_back(std::make_shared<NewBicycleNode<BoxConstraints<6, 2>>>(
            L, dt, 0.001, goal_list_fast[i], Q_fast, R_fast, constraints_obj));
    }

    // 定义初始状态: [x0, y0, θ0, δ0, v0, a0]
    Eigen::Matrix<double, 6,1> init_state;
    init_state << 0.0, 0.0, 0.0, 0.0, v, 0.0;  // 从原点出发，速度为 v

    // 定义障碍物（用于求解器内部的障碍物表示，虽然这里不直接使用）
    // left_car: 2×4 矩阵，每列表示一个障碍物顶点 [x; y]
    Eigen::Matrix<double, 2, 4> left_car;
    left_car << 32, 32, 28, 28,  // 障碍物 x 坐标
                14, 16, 16, 14;  // 障碍物 y 坐标（矩形框）
    std::vector<Eigen::Matrix<double, 2, 4>> left_obs;
    std::vector<Eigen::Matrix<double, 2, 4>> right_obs;
    left_obs.push_back(left_car);
    right_obs.clear();

    // 创建增广拉格朗日 iLQR 求解器
    NewALILQR<6,2> solver(ilqr_node_list, init_state, left_obs, right_obs);

    // 执行优化
    // 参数: (外层最大迭代次数, 内层最大迭代次数, 约束违反容忍度)
    solver.optimize(50, 100, 1e-3);

    // 输出控制序列结果
    std::cout << "\n========== 盒式约束优化结果 ==========" << std::endl;
    for(int i = 0; i < num_points - 1; ++i) {
        std::cout << "u_result " << solver.get_u_list().col(i).transpose() << std::endl;
    }

    // 输出状态轨迹结果
    for(int i = 0; i < num_points - 1; ++i) {
        std::cout << "x_result " << solver.get_x_list().col(i).transpose() << std::endl;
    }

    // ========== 测试二：使用二次约束的轨迹优化（圆形障碍物避障） ==========

    // 定义约束维度（包含 1 个圆形障碍物约束 + 4 个控制界限约束）
    constexpr  int constraint_dim = 5;

    // 二次约束的系数矩阵
    // 约束形式: Q[i]*x^T*x + A[i]*x + B[i]*u + C[i] <= 0
    Eigen::Matrix<double, constraint_dim, 6> A;  // 状态的一次项系数矩阵
    Eigen::Matrix<double, constraint_dim, 2> B;  // 控制的一次项系数矩阵
    Eigen::Matrix<double, constraint_dim, 1> C;  // 常数项向量
    A.setZero();
    B.setZero();
    C.setZero();

    // 定义控制约束（第 1-4 个约束）
    // 约束 1-2: δ_rate 的上下界约束
    // 约束 3-4: a_rate 的上下界约束
    B << 0, 0,      // 约束 0：圆形障碍物（稍后设置）
         1, 0,      // 约束 1：δ_rate <= 0.2
         0, 1,      // 约束 2：a_rate <= 1
        -1, 0,      // 约束 3：-δ_rate <= 0.2 (即 δ_rate >= -0.2)
         0, -1;     // 约束 4：-a_rate <= 1 (即 a_rate >= -1)
    C << 0,         // 约束 0：圆形障碍物（稍后设置）
        -0.2,       // 约束 1：δ_rate - 0.2 <= 0
        -1,         // 约束 2：a_rate - 1 <= 0
        -0.2,       // 约束 3：-δ_rate - 0.2 <= 0
        -1;         // 约束 4：-a_rate - 1 <= 0

    // 初始化二次项系数矩阵数组
    std::array<Eigen::Matrix<double, 6, 6>, constraint_dim> Q;
    for (int i = 0; i < constraint_dim; ++i) {
        Q[i] = Eigen::Matrix<double, 6, 6>::Zero();
    }

    // 生成圆形障碍物约束（第 0 个约束）
    // 障碍物位置: (20, 12), 半径: 4m
    auto ans = GenerateCycleEquations(20.0, 12, 4.0, 6);

    // 设置圆形约束的系数
    Q[0] = std::get<0>(ans);            // 二次项系数矩阵
    C(0, 0) = (std::get<2>(ans)).value();  // 常数项
    A.row(0) = std::get<1>(ans);        // 一次项系数

    // 创建二次约束对象
    QuadraticConstraints<6, 2, constraint_dim> quad_constrants(Q, A, B, C);

    // 创建使用二次约束的 iLQR 节点列表
    std::vector<std::shared_ptr<NewILQRNode<6, 2>>> q_ilqr_node_list;
    q_ilqr_node_list.clear();

    // 为每个时间步创建带二次约束的自行车模型节点
    for (int i = 0; i <= num_points; ++i) {
        q_ilqr_node_list.push_back(std::make_shared<NewBicycleNode<QuadraticConstraints<6, 2, constraint_dim>>>(
            L, dt, 0.001, goal_list_fast[i], Q_fast, R_fast, quad_constrants));
    }

    // 创建二次约束求解器（不使用障碍物表示，约束已通过 QuadraticConstraints 定义）
    NewALILQR<6,2> q_solver(q_ilqr_node_list, init_state);

    // 执行优化
    // 参数: (外层最大迭代次数, 内层最大迭代次数, 约束违反容忍度)
    q_solver.optimize(30, 100, 1e-3);

    // 输出控制序列结果
    std::cout << "\n========== 二次约束优化结果 (含圆形障碍物) ==========" << std::endl;
    for(int i = 0; i < num_points - 1; ++i) {
        std::cout << "q u_result " << q_solver.get_u_list().col(i).transpose() << std::endl;
    }

    // 输出状态轨迹结果
    for(int i = 0; i < num_points - 1; ++i) {
        std::cout << "q x_result " << q_solver.get_x_list().col(i).transpose() << std::endl;
    }

    std::cout << "\n========== 测试完成 ==========" << std::endl;

    return 0;
}