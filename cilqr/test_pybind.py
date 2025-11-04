"""
iLQR Python 绑定测试脚本 - 完整动力学自行车模型

本脚本演示如何使用 C++ 实现的 iLQR 求解器（通过 pybind11 绑定）
进行车辆轨迹优化，包括：
1. 盒式约束优化（仅状态和控制边界）
2. 二次约束优化（包含圆形障碍物避障）

状态向量: [x, y, theta, delta, v, a]
  - x, y: 车辆位置 (m)
  - theta: 航向角 (rad)
  - delta: 前轮转角 (rad)
  - v: 车辆速度 (m/s)
  - a: 车辆加速度 (m/s^2)

控制向量: [delta_rate, a_rate]
  - delta_rate: 前轮转角变化率 (rad/s)
  - a_rate: 加速度变化率 (m/s^3, jerk)
"""

import sys
import numpy as np
# 添加 C++ 编译生成的 Python 绑定模块路径
sys.path.append("/home/pnc/Documents/github/ilqr/cilqr/al_ilqr_cpp/bazel-bin")
import ilqr_pybind
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 配置中文字体支持
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']  # 或 'Noto Sans CJK SC'
plt.rcParams['axes.unicode_minus'] = False  # 避免负号显示为方块

def generate_s_shape_goal_full(v, dt, num_points):
    """
    生成 S 形参考轨迹（完整动力学状态）

    使用正弦函数生成一条平滑的 S 形曲线，并计算对应的完整动力学状态。

    参数:
        v (float): 期望速度 (m/s)
        dt (float): 时间步长 (s)
        num_points (int): 轨迹点数量

    返回:
        list: 包含 (num_points+1) 个状态向量的列表
              每个状态为 [x, y, theta, delta, v, a]

    轨迹方程:
        x(t) = v * t                    # 匀速前进
        y(t) = 50 * sin(0.1 * t)        # 正弦横摆

    物理量计算:
        - 航向角 theta 由速度方向计算: arctan2(dy/dt, dx/dt)
        - 曲率 kappa 由二阶导数计算: (dx*ddy - dy*ddx) / (dx^2 + dy^2)^1.5
        - 前轮转角 delta 通过 Ackermann 转向模型近似: arctan(kappa * L)
    """
    goals = []
    for i in range(num_points + 1):
        t = i * dt

        # 位置: x 方向匀速运动，y 方向正弦摆动
        x = v * t
        y = 50 * np.sin(0.1 * t)

        # 一阶导数 (速度方向)
        dx = v                          # x 方向速度恒定
        dy = 50 * 0.1 * np.cos(0.1 * t) # y 方向速度

        # 二阶导数 (加速度方向)
        ddx = 0                                 # x 方向加速度为 0
        ddy = -50 * 0.1 * 0.1 * np.sin(0.1 * t) # y 方向加速度

        # 计算航向角: 速度矢量的方向
        theta = np.arctan2(dy, dx)

        # 计算曲率: 路径的弯曲程度
        # 公式: kappa = (dx*d²y - dy*d²x) / (dx² + dy²)^(3/2)
        curvature = (dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** 1.5

        # 根据曲率计算前轮转角 (假设车辆轴距为 1.0m)
        delta = np.arctan(curvature * 1.0)

        # 组装完整状态向量: [x, y, theta, delta, v_desire, a_desire]
        goal_state = np.array([x, y, theta, delta, v, 0])
        goals.append(goal_state)

    return goals


def generate_cycle_equations(centre_x, centre_y, r, x_dims):
    """
    生成圆形障碍物的二次约束矩阵

    将圆形障碍物表示为二次不等式约束的形式:
        x^T * Q * x + A^T * x + C <= 0

    对于圆: (x - cx)^2 + (y - cy)^2 <= r^2
    展开为: x^2 + y^2 - 2*cx*x - 2*cy*y + (cx^2 + cy^2 - r^2) <= 0

    参数:
        centre_x (float): 圆心 x 坐标 (m)
        centre_y (float): 圆心 y 坐标 (m)
        r (float): 圆的半径 (m)
        x_dims (int): 状态向量维度（通常为 6）

    返回:
        tuple: (Q, A, C) 三个约束矩阵
            - Q: (x_dims, x_dims) 二次项系数矩阵
            - A: (1, x_dims) 一次项系数向量
            - C: (1, 1) 常数项

    约束形式: x^T * Q * x + A^T * x + C <= 0
    实际表示: -(x^2 + y^2) + 2*cx*x + 2*cy*y + (cx^2 + cy^2 - r^2) <= 0
    等价于: (x - cx)^2 + (y - cy)^2 >= r^2  (障碍物外部)
    """
    # 初始化约束矩阵
    Q = np.zeros((x_dims, x_dims))  # 二次项系数
    A = np.zeros((1, x_dims))       # 一次项系数
    C = np.zeros((1, 1))            # 常数项

    # 设置常数项: cx^2 + cy^2 - r^2
    C[0, 0] = r * r - centre_x * centre_x - centre_y * centre_y

    # 设置二次项: -x^2 - y^2 (负号表示约束为"外部区域")
    Q[0, 0] = -1.0  # x^2 的系数
    Q[1, 1] = -1.0  # y^2 的系数

    # 设置一次项: +2*cx*x + 2*cy*y
    A[0, 0] = 2 * centre_x  # x 的系数
    A[0, 1] = 2 * centre_y  # y 的系数

    return Q, A, C


# ============================================================================
# 第一部分: 生成参考轨迹
# ============================================================================

print("=" * 60)
print("生成 S 形参考轨迹...")
print("=" * 60)

# 车辆和仿真参数
v = 10          # 期望速度: 10 m/s
dt = 0.1        # 时间步长: 0.1 秒
L = 3           # 车辆轴距: 3 米
k = 0.001       # 正则化系数（用于数值稳定性）
num_points = 30 # 轨迹点数量
horizon = 30    # 优化时域长度

# 生成 S 形参考轨迹
goal_list_full = generate_s_shape_goal_full(v, dt, num_points)

# 提取参考轨迹的 x, y 坐标用于可视化
goal_x = [goal[0] for goal in goal_list_full]
goal_y = [goal[1] for goal in goal_list_full]

print(f"参考轨迹点数: {len(goal_list_full)}")
print(f"起点: x={goal_x[0]:.2f}, y={goal_y[0]:.2f}")
print(f"终点: x={goal_x[-1]:.2f}, y={goal_y[-1]:.2f}")


# ============================================================================
# 第二部分: 配置优化器参数和约束（盒式约束）
# ============================================================================

print("\n" + "=" * 60)
print("配置优化器参数 - 盒式约束优化")
print("=" * 60)

# 状态和控制维度
state_dim = 6    # 状态: [x, y, theta, delta, v, a]
control_dim = 2  # 控制: [delta_rate, a_rate]

# 代价函数权重矩阵
# Q: 状态误差权重 (相对于参考轨迹的偏差惩罚)
#    [x, y, theta, delta, v, a]
Q = np.diag([1e-1, 1e-1, 1e-0, 1e-9, 1e-6, 1e-6]) * 1e3
# Q[0,0]=100: x 位置误差权重
# Q[1,1]=100: y 位置误差权重
# Q[2,2]=1000: 航向角误差权重（最重要）
# Q[3,3]=0.001: 前轮转角误差权重
# Q[4,4]=1: 速度误差权重
# Q[5,5]=1: 加速度误差权重

# R: 控制输入权重 (控制输入平滑性惩罚)
#    [delta_rate, a_rate]
R = np.diag([1, 1]) * 1e2
# R[0,0]=100: 转角变化率权重
# R[1,1]=100: 加速度变化率权重

print(f"状态权重矩阵 Q 对角元素: {np.diag(Q)}")
print(f"控制权重矩阵 R 对角元素: {np.diag(R)}")

# 盒式约束: 状态和控制的上下界
state_min = np.array([-1000, -1000, -2 * np.pi, -10, -100, -10])
#                     [x_min, y_min, theta_min, delta_min, v_min, a_min]
state_max = np.array([1000, 1000, 2 * np.pi, 10, 100, 10])
#                     [x_max, y_max, theta_max, delta_max, v_max, a_max]

control_min = np.array([-0.2, -1])  # [delta_rate_min, a_rate_min]
control_max = np.array([0.2, 1])    # [delta_rate_max, a_rate_max]

print(f"状态约束: [{state_min[3]:.2f}, {state_max[3]:.2f}] rad (转角)")
print(f"控制约束: [{control_min[0]:.2f}, {control_max[0]:.2f}] rad/s (转角率)")

# 创建盒式约束对象 (6维状态, 2维控制)
constraints = ilqr_pybind.BoxConstraints6_2(
    state_min, state_max, control_min, control_max
)


# ============================================================================
# 第三部分: 构建动力学节点并执行优化（盒式约束）
# ============================================================================

print("\n" + "=" * 60)
print("构建动力学节点列表...")
print("=" * 60)

# 为每个时间步创建一个动力学节点
# 每个节点包含: 车辆模型、参考状态、代价矩阵、约束
ilqr_nodes_list = []
for i in range(horizon + 1):
    # NewBicycleNodeBoxConstraints6_2: 自行车动力学模型 + 盒式约束
    # 参数: (轴距, 时间步长, 正则化系数, 参考状态, Q, R, 约束)
    node = ilqr_pybind.NewBicycleNodeBoxConstraints6_2(
        L, dt, k, goal_list_full[i], Q, R, constraints
    )
    ilqr_nodes_list.append(node)

print(f"动力学节点数量: {len(ilqr_nodes_list)}")

# 初始状态: 车辆从原点出发，初始速度为 v
init_state = np.array([0, 0, 0, 0, v, 0])
#                      [x, y, θ, δ, v, a]
print(f"初始状态: {init_state}")

# 创建增广拉格朗日 iLQR 求解器
# NewALILQR6_2: 6维状态, 2维控制
al_ilqr = ilqr_pybind.NewALILQR6_2(ilqr_nodes_list, init_state)

# 设置优化参数
max_outer_iter = 50     # 外层迭代 (增广拉格朗日法更新 λ 和 μ)
max_inner_iter = 100    # 内层迭代 (iLQR 优化)
max_violation = 1e-4    # 约束违反容忍度

print("\n" + "=" * 60)
print("开始优化 - 盒式约束...")
print("=" * 60)
print(f"最大外层迭代: {max_outer_iter}")
print(f"最大内层迭代: {max_inner_iter}")
print(f"约束违反容忍: {max_violation}")

# 执行优化
al_ilqr.optimize(max_outer_iter, max_inner_iter, max_violation)

# 获取优化后的状态和控制序列
x_list = al_ilqr.get_x_list()  # shape: (6, horizon+1)
u_list = al_ilqr.get_u_list()  # shape: (2, horizon)

# 提取 x, y 坐标用于可视化
plot_x = x_list[0, :]  # x 坐标序列
plot_y = x_list[1, :]  # y 坐标序列

print("优化完成！")
print(f"优化轨迹长度: {len(plot_x)} 个点")


# ============================================================================
# 第四部分: 添加障碍物约束并重新优化（二次约束）
# ============================================================================

print("\n" + "=" * 60)
print("配置障碍物约束 - 二次约束优化")
print("=" * 60)

# 准备二次约束容器
# 这里我们定义 5 个约束:
# 1. 圆形障碍物 (二次约束)
# 2-5. 线性边界约束 (可选，这里用于演示)
Q_list = []
for i in range(5):
    Q_signal = np.zeros((6, 6))
    Q_list.append(Q_signal)

# 线性约束矩阵
# 约束形式: B * u + A * x <= C
# 这里定义 5 个线性约束的参数
A = np.zeros((5, 6))  # 状态系数矩阵
B = np.array([
    [0, 0],   # 第 1 个约束 (将被圆形障碍物覆盖)
    [1, 0],   # 第 2 个约束: 控制输入 u[0] 的约束
    [0, 1],   # 第 3 个约束: 控制输入 u[1] 的约束
    [-1, 0],  # 第 4 个约束: 控制输入 -u[0] 的约束
    [0, -1]   # 第 5 个约束: 控制输入 -u[1] 的约束
])
C = np.array([
    [0],      # 第 1 个约束的右端项 (将被圆形障碍物覆盖)
    [-0.4],   # u[0] <= -0.4
    [-1],     # u[1] <= -1
    [-0.4],   # -u[0] <= -0.4
    [-1]      # -u[1] <= -1
])

# 定义圆形障碍物
circle_x = 30   # 圆心 x 坐标 (m)
circle_y = 11   # 圆心 y 坐标 (m)
circle_r = 6    # 圆的半径 (m)

print(f"圆形障碍物: 圆心=({circle_x}, {circle_y}), 半径={circle_r}m")

# 生成圆形障碍物的二次约束矩阵
Qc, Ac, Cc = generate_cycle_equations(circle_x, circle_y, circle_r, 6)

# 将圆形障碍物设置为第 1 个约束
Q_list[0] = Qc      # 二次项矩阵
C[0, 0] = Cc.item() # 常数项
A[0, :] = Ac        # 一次项向量

# 创建二次约束对象
# QuadraticConstraints6_2_5: 6维状态, 2维控制, 5个约束
quadratic_constraints = ilqr_pybind.QuadraticConstraints6_2_5(Q_list, A, B, C)

print("二次约束配置完成")

# 使用二次约束重新构建动力学节点列表
quadratic_ilqr_nodes_list = []
for i in range(horizon + 1):
    # NewBicycleNodeQuadraticConstraints6_2_5: 自行车模型 + 二次约束
    node = ilqr_pybind.NewBicycleNodeQuadraticConstraints6_2_5(
        L, dt, k, goal_list_full[i], Q, R, quadratic_constraints
    )
    quadratic_ilqr_nodes_list.append(node)

print(f"二次约束动力学节点数量: {len(quadratic_ilqr_nodes_list)}")

# 创建带有障碍物约束的求解器
q_al_ilqr = ilqr_pybind.NewALILQR6_2(quadratic_ilqr_nodes_list, init_state)

print("\n" + "=" * 60)
print("开始优化 - 二次约束 (包含障碍物避障)...")
print("=" * 60)

# 执行优化
q_al_ilqr.optimize(max_outer_iter, max_inner_iter, max_violation)

# 获取优化后的状态和控制序列
q_x_list = q_al_ilqr.get_x_list()
q_u_list = q_al_ilqr.get_u_list()

# 提取 x, y 坐标用于可视化
q_plot_x = q_x_list[0, :]
q_plot_y = q_x_list[1, :]

print("优化完成！")
print(f"避障轨迹长度: {len(q_plot_x)} 个点")


# ============================================================================
# 第五部分: 可视化对比三条轨迹
# ============================================================================

print("\n" + "=" * 60)
print("生成可视化...")
print("=" * 60)

plt.figure(figsize=(10, 6))
ax = plt.gca()

# 绘制圆形障碍物
circle = patches.Circle(
    (circle_x, circle_y), circle_r,
    edgecolor='green',
    facecolor='lightblue',
    alpha=0.5,
    linewidth=2,
    label='circle obstacle'
)
ax.add_patch(circle)

# 绘制三条轨迹
plt.plot(plot_x, plot_y,
         label='Optimized State Trajectory (Full)',
         c='b', marker='o', markersize=4, linewidth=2) # 无障碍物约束优化轨迹 (盒式约束)

plt.plot(goal_x, goal_y,
         label='init State Trajectory (Full)',
         c='r', marker='o', markersize=4, linewidth=2)# 参考轨迹 (S形曲线)

plt.plot(q_plot_x, q_plot_y,
         label='obs (Full)',
         c='g', marker='o', markersize=4, linewidth=2) # 避障优化轨迹 (二次约束)

# 标记起点和终点
plt.plot(init_state[0], init_state[1], 'ko', markersize=10, label='start')
plt.plot(goal_x[-1], goal_y[-1], 'k*', markersize=15, label='goal')

# 设置坐标轴
ax.set_aspect('equal')  # 确保 x, y 轴比例相同
plt.xlabel('X Position (m)', fontsize=12)
plt.ylabel('Y Position (m)', fontsize=12)
plt.title('iLQR Trajectory Optimization Comparison - Full Dynamic Bicycle Model', fontsize=14, fontweight='bold')
plt.legend(fontsize=10, loc='best')
plt.grid(True, alpha=0.3)
plt.axis('equal')

print("\n图例说明:")
print("  🔴 红色虚线: 理想的 S 形参考轨迹")
print("  🔵 蓝色实线: 仅考虑盒式约束的优化轨迹 (可能穿过障碍物)")
print("  🟢 绿色实线: 考虑障碍物约束的优化轨迹 (绕开障碍物)")
print("  ⭕ 圆形区域: 需要避开的障碍物")

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)
