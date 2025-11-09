"""
矩阵符号计算模块
用于计算二次型表达式的符号展开
常用于iLQR（迭代线性二次调节器）中的代价函数计算
"""

# 导入sympy库中的矩阵符号、转置和符号定义功能
from sympy import MatrixSymbol, Transpose, symbols
import sympy

# 定义整数符号 n 和 m，分别表示状态维度和控制输入维度
# integer=True: 确保是整数类型
# positive=True: 确保是正数
n, m = symbols('n m', integer=True, positive=True)

# 定义状态转移矩阵 A，维度为 n×n
# 表示线性系统的状态转移矩阵
A = MatrixSymbol('A', n, n)

# 定义控制输入矩阵 B，维度为 n×m
# 表示控制输入对状态的影响
B = MatrixSymbol('B', n, m)

# 定义权重矩阵 P，维度为 n×n
# 通常用于代价函数中的状态权重矩阵
P = MatrixSymbol('P', n, n)

# 定义状态向量 x，维度为 n×1
# 表示系统的当前状态
x = MatrixSymbol('x', n, 1)

# 定义控制输入向量 u，维度为 m×1
# 表示系统的控制输入
u = MatrixSymbol('u', m, 1)

# 计算二次型表达式: (1/2) * (Ax + Bu)^T * P * (Ax + Bu)
# 这是典型的二次代价函数形式，常用于最优控制问题
# Transpose(A*x + B*u): 计算 (Ax + Bu) 的转置
# P * (A*x + B*u): 与权重矩阵 P 相乘
# 整体除以2是为了后续求导方便（消除系数）
expr = (Transpose(A*x + B*u) * P * (A*x + B*u))/2

# 展开表达式并打印结果
# expand() 函数将表达式展开为多项式形式，便于查看和进一步计算
print(sympy.expand(expr))  # 展开

# result is:
# (1/2)*u.T*B.T*P*A*x + (1/2)*u.T*B.T*P*B*u + (1/2)*x.T*A.T*P*A*x + (1/2)*x.T*A.T*P*B*u