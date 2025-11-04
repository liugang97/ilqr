# ~/.gdbinit
set step-mode off

# STL 模板跳过
skip file /usr/include/c++/*/bits/*
skip file /usr/include/c++/*/ext/*
skip file /usr/include/c++/*/type_traits
skip file /usr/include/c++/*/utility

# Eigen 所有实现（系统 / external / bazel）
skip file */eigen/Eigen/src/*
skip file */external/eigen/Eigen/src/*
skip file */bazel-out/*/external/eigen/Eigen/src/*
skip file */execroot/*/external/eigen/Eigen/src/*

# Boost (可选)
skip file */boost/*

python
import sys
sys.path.insert(0, '/home/pnc/Documents/github/MyDocs/my_scripts/gdb/')
from printers import register_eigen_printers
register_eigen_printers(None)
end
