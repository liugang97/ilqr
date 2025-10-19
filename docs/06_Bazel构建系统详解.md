# Bazel 构建系统详解 - iLQR 项目实战指南

> 本文档深入解析 `cilqr/al_ilqr_cpp` 项目如何使用 Bazel 构建系统，涵盖配置、依赖管理、编译优化等核心技术点。

---

## 目录

1. [Bazel 简介](#1-bazel-简介)
2. [项目中的 Bazel 架构](#2-项目中的-bazel-架构)
3. [核心配置文件详解](#3-核心配置文件详解)
4. [依赖管理机制](#4-依赖管理机制)
5. [构建规则详解](#5-构建规则详解)
6. [编译优化配置](#6-编译优化配置)
7. [完整构建流程](#7-完整构建流程)
8. [常用命令与技巧](#8-常用命令与技巧)
9. [问题排查与调试](#9-问题排查与调试)

---

## 1. Bazel 简介

### 1.1 什么是 Bazel

**Bazel** 是 Google 开源的构建和测试工具，专为大规模多语言项目设计。本项目使用 Bazel 的核心原因：

- **增量构建**: 只重新编译修改的部分，大幅提升构建速度
- **精确依赖管理**: 显式声明依赖关系，避免隐式依赖导致的问题
- **可重现构建**: 相同输入保证相同输出，适合 CI/CD 环境
- **跨平台支持**: 统一的构建脚本适用于 Linux、macOS、Windows
- **多语言支持**: 同时构建 C++、Python、Java 等多种语言

### 1.2 项目选用 Bazel 的原因

| 特性 | CMake | Bazel | 项目需求 |
|------|-------|-------|---------|
| C++/Python 混合构建 | 需要额外配置 | 原生支持 | ✅ 必需 |
| pybind11 集成 | 手动配置复杂 | 有官方规则 | ✅ 必需 |
| Eigen 依赖管理 | 需要 Find*.cmake | HTTP archive 自动下载 | ✅ 简化依赖 |
| 编译优化 | 分散在多处 | 统一配置 (.bazelrc) | ✅ 提高性能 |
| 增量构建速度 | 较慢 | 极快 | ✅ 提升效率 |

---

## 2. 项目中的 Bazel 架构

### 2.1 目录结构

```
cilqr/al_ilqr_cpp/
├── WORKSPACE                   # 工作空间定义 (外部依赖)
├── MODULE.bazel                # Bazel 模块定义 (新版本方式)
├── MODULE.bazel.lock           # 模块依赖锁定文件
├── .bazelrc                    # Bazel 配置文件 (编译选项)
├── BUILD                       # 根目录构建规则
├── eigen.BUILD                 # Eigen 自定义构建规则
│
├── model/
│   └── BUILD                   # 车辆模型构建规则
├── constraints/
│   └── BUILD                   # 约束类构建规则
│
└── bazel-*                     # Bazel 生成的符号链接 (自动生成)
    ├── bazel-bin/              # 编译产物输出目录
    ├── bazel-out/              # 中间编译产物
    ├── bazel-al_ilqr_cpp/      # 指向 execroot
    └── bazel-testlogs/         # 测试日志
```

### 2.2 构建目标依赖图

```
ilqr_pybind.so (Python 扩展模块)
    │
    ├── ilqr_pybind.cc (源文件)
    ├── //:new_al_ilqr (C++ 库)
    │   └── model:new_ilqr_node
    │       └── constraints:box_constraints
    │           └── constraints:linear_constraints
    │               └── constraints:constraints
    ├── //model:node_bind (绑定头文件)
    │   ├── model:new_bicycle_node
    │   └── model:new_lat_bicycle_node
    ├── //constraints:constraints_bind
    │
    └── 外部依赖
        ├── @eigen (线性代数库)
        ├── @pybind11 (C++/Python 绑定库)
        └── @local_config_python (Python 配置)
```

---

## 3. 核心配置文件详解

### 3.1 WORKSPACE - 外部依赖声明

**文件**: `cilqr/al_ilqr_cpp/WORKSPACE`

```python
workspace(name = "al_ilqr_project")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# ============================================================
# 1. Eigen 线性代数库 (v3.3.7)
# ============================================================
http_archive(
    name = "eigen",
    build_file = "//:eigen.BUILD",    # 使用自定义构建文件
    sha256 = "a8d87c8df67b0404e97bcef37faf3b140ba467bc060e2b883192165b319cea8d",
    strip_prefix = "eigen-git-mirror-3.3.7",
    urls = [
        # 备用镜像源
        "https://apollo-system.cdn.bcebos.com/archive/6.0/3.3.7.tar.gz",
        "https://github.com/eigenteam/eigen-git-mirror/archive/3.3.7.tar.gz",
    ],
)

# ============================================================
# 2. pybind11_bazel - pybind11 构建规则
# ============================================================
http_archive(
    name = "pybind11_bazel",
    strip_prefix = "pybind11_bazel-b162c7c88a253e3f6b673df0c621aca27596ce6b",
    urls = ["https://github.com/pybind/pybind11_bazel/archive/b162c7c.zip"],
    sha256 = "b72c5b44135b90d1ffaba51e08240be0b91707ac60bea08bb4d84b47316211bb",
)

# ============================================================
# 3. pybind11 库 (v2.13.6)
# ============================================================
http_archive(
    name = "pybind11",
    build_file = "@pybind11_bazel//:pybind11.BUILD",  # 使用官方构建文件
    strip_prefix = "pybind11-2.13.6",
    urls = ["https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.tar.gz"],
    sha256 = "e08cb87f4773da97fa7b5f035de8763abc656d87d5773e62f6da0587d1f0ec20",
)

# ============================================================
# 4. Python 环境自动配置
# ============================================================
load("@pybind11_bazel//:python_configure.bzl", "python_configure")
python_configure(name = "local_config_python")
```

#### 关键技术点

**① http_archive 规则**

```python
http_archive(
    name = "依赖名称",           # Bazel 中引用的名称 (如 @eigen)
    urls = ["下载 URL"],         # 支持多个备用 URL
    sha256 = "校验和",           # 文件完整性校验 (可选但推荐)
    strip_prefix = "前缀路径",   # 解压后去除的目录前缀
    build_file = "构建文件路径", # 自定义 BUILD 文件 (对于无官方支持的库)
)
```

**② 为什么 Eigen 需要自定义 build_file?**

Eigen 官方仓库没有 Bazel 支持，因此需要手动编写 `eigen.BUILD` 文件来定义如何构建它。

**③ python_configure() 的作用**

自动检测系统 Python 环境：
- 查找 Python 解释器路径
- 检测 `Python.h` 头文件位置 (`/usr/include/python3.x`)
- 查找 `libpython*.so` 库文件
- 生成 `@local_config_python` 仓库供构建使用

### 3.2 MODULE.bazel - 新版依赖管理

**文件**: `cilqr/al_ilqr_cpp/MODULE.bazel`

```python
bazel_dep(name = "pybind11_bazel", version = "2.11.1")
```

#### Bazel 版本演进

| 方式 | 文件 | 优势 | 劣势 |
|------|------|------|------|
| **WORKSPACE (旧版)** | WORKSPACE | 灵活，支持复杂配置 | 依赖版本冲突难解决 |
| **MODULE (新版)** | MODULE.bazel | 自动解决依赖冲突 | 功能尚未完全覆盖 |

**本项目采用混合方式**:
- `MODULE.bazel`: 声明 pybind11_bazel (新版模块仓库中可用)
- `WORKSPACE`: 管理 Eigen、pybind11 等 (暂无模块支持)

### 3.3 .bazelrc - 编译配置

**文件**: `cilqr/al_ilqr_cpp/.bazelrc`

```bash
# ============================================================
# Debug 模式配置
# ============================================================
build:debug --compilation_mode=dbg

# ============================================================
# 优化模式配置 (生产环境)
# ============================================================
build:opt --copt=-O3                  # C 编译器优化级别 3
build:opt --copt=-march=native        # 针对本地 CPU 指令集优化
build:opt --cxxopt=-O3                # C++ 编译器优化级别 3
build:opt --cxxopt=-march=native      # 启用 AVX/SSE 等 SIMD 指令
build:opt --linkopt=-O3               # 链接时优化 (LTO)

# ============================================================
# Python 路径配置 (特定环境)
# ============================================================
build --python_path=/home/vincent/.conda/envs/py39/bin/python
```

#### 使用方式

```bash
# 默认模式 (fastbuild)
bazel build //:ilqr_pybind.so

# 使用优化模式
bazel build --config=opt //:ilqr_pybind.so

# 使用调试模式
bazel build --config=debug //:ilqr_pybind.so
```

#### 编译标志详解

| 标志 | 作用 | 性能影响 | 副作用 |
|------|------|---------|--------|
| `-O3` | 最高级别优化 | +30-50% | 编译时间增加 |
| `-march=native` | 使用本地 CPU 指令 | +10-20% | 二进制不可移植 |
| `-faligned-new` | C++17 对齐分配 | - | 兼容性要求 |
| `-DEIGEN_VECTORIZE` | 启用 Eigen 向量化 | +20-40% | 需要支持 SSE/AVX |

**⚠️ 注意**: `-march=native` 生成的二进制文件只能在相同或更高级 CPU 上运行！

### 3.4 eigen.BUILD - 自定义 Eigen 构建

**文件**: `cilqr/al_ilqr_cpp/eigen.BUILD`

```python
load("@rules_cc//cc:defs.bzl", "cc_library")

# 许可证声明
licenses([
    "reciprocal",  # MPL2
    "notice",      # Portions BSD
])

# 定义需要包含的头文件路径
EIGEN_FILES = [
    "Eigen/**",                          # 核心模块
    "unsupported/Eigen/CXX11/**",        # C++11 特性
    "unsupported/Eigen/FFT",             # 快速傅里叶变换
    "unsupported/Eigen/**",
    "unsupported/Eigen/MatrixFunctions",  # 矩阵函数
    "unsupported/Eigen/SpecialFunctions", # 特殊函数
    # ... 更多模块
]

# 排除测试文件
EIGEN_EXCLUDE_FILES = [
    "Eigen/src/Core/arch/AVX/PacketMathGoogleTest.cc",
]

# 使用 glob 收集所有头文件
EIGEN_MPL2_HEADER_FILES = glob(
    EIGEN_FILES,
    exclude = EIGEN_EXCLUDE_FILES,
)

# 定义 cc_library 目标
cc_library(
    name = "eigen",
    hdrs = EIGEN_MPL2_HEADER_FILES,   # 头文件列表
    includes = ["."],                 # 添加到 include 路径
    visibility = ["//visibility:public"],  # 公开可见
)
```

#### 关键概念

**① Header-only 库**

Eigen 是纯头文件库，无需编译 `.cc` 文件，因此：
- `srcs = []` (无源文件)
- `hdrs = [...]` (仅头文件)
- `includes = ["."]` 将当前目录添加到编译器 `-I` 参数

**② glob() 函数**

```python
glob(["pattern"], exclude=["pattern"])
```

动态匹配文件，支持通配符 `**` (递归) 和 `*` (单层)。

**③ visibility 控制**

```python
visibility = ["//visibility:public"]   # 所有包可见
visibility = ["//visibility:private"]  # 仅当前包可见
visibility = ["//model:__pkg__"]       # 仅 model 包可见
```

---

## 4. 依赖管理机制

### 4.1 外部依赖加载流程

```
执行 bazel build
    ↓
读取 WORKSPACE 文件
    ↓
发现 http_archive(name = "eigen", urls = [...])
    ↓
检查本地缓存: ~/.cache/bazel/_bazel_<user>/cache/repos/
    ↓
    ├── 已缓存 → 直接使用
    └── 未缓存 → 下载并解压
            ↓
        计算 SHA256 校验和
            ↓
        校验通过 → 应用 strip_prefix
            ↓
        读取 build_file 指定的 eigen.BUILD
            ↓
        在 execroot 中创建符号链接: external/eigen/
            ↓
        后续构建可引用 @eigen//:eigen
```

### 4.2 依赖引用语法

| 引用方式 | 示例 | 含义 |
|---------|------|------|
| 本地库 | `//model:new_ilqr_node` | 引用 `model/BUILD` 中的 `new_ilqr_node` 目标 |
| 外部库 | `@eigen//:eigen` | 引用 WORKSPACE 中定义的 `eigen` 仓库的 `eigen` 目标 |
| 绝对路径 | `//:ilqr_pybind.so` | 引用根目录 BUILD 文件中的目标 |
| 相对路径 | `:new_al_ilqr` | 引用当前 BUILD 文件中的目标 |

### 4.3 依赖传递规则

```python
# constraints/BUILD
cc_library(
    name = "constraints",
    hdrs = ["constraints.h"],
    deps = ["@eigen"],  # 依赖 Eigen
)

cc_library(
    name = "box_constraints",
    hdrs = ["box_constraints.h"],
    deps = [":linear_constraints"],  # 依赖同包的 linear_constraints
)

# 根 BUILD
cc_library(
    name = "new_al_ilqr",
    hdrs = ["new_al_ilqr.h"],
    deps = [
        "//constraints:box_constraints",  # 自动传递 Eigen 依赖
    ],
)
```

**关键特性**:
- **传递依赖自动处理**: `new_al_ilqr` 间接依赖 `@eigen`，无需显式声明
- **去重优化**: Bazel 自动去除重复依赖

---

## 5. 构建规则详解

### 5.1 cc_library - C++ 库

**基本用法** (文件: `model/BUILD:57-69`)

```python
cc_library(
    name = "new_ilqr_node",           # 目标名称
    hdrs = ["new_ilqr_node.h"],       # 公开头文件 (其他库可引用)
    srcs = ["new_ilqr_node.cc"],      # 源文件 (可选,本例为纯头文件库)

    copts = [                          # 编译选项 (compile options)
        "-O3",
        "-march=native",
        "-DEIGEN_VECTORIZE"
    ],

    deps = [                           # 依赖
        "@eigen",
        "//constraints:box_constraints",
    ],

    visibility = ["//visibility:public"],  # 可见性
)
```

#### hdrs vs srcs 的区别

| 属性 | 用途 | 编译行为 | 示例 |
|------|------|---------|------|
| `hdrs` | 接口定义 (头文件) | 被依赖方可见 | `.h`, `.hpp` |
| `srcs` | 实现代码 | 仅当前目标编译 | `.cc`, `.cpp` |

**最佳实践**:
- 模板类放在 `hdrs` (需要在使用处实例化)
- 函数实现放在 `srcs` (减少重复编译)

### 5.2 cc_binary - 可执行文件

**示例** (文件: `BUILD:51-61`)

```python
cc_binary(
    name = "test_new_al_ilqr",        # 生成可执行文件名
    srcs = ["test_new_al_ilqr.cc"],   # 必须包含 main() 函数
    deps = [
        ":new_al_ilqr",
        "//model:new_bicycle_node",
        "//constraints:box_constraints",
    ],
    copts = ["-O3", "-march=native", "-faligned-new", "-DEIGEN_VECTORIZE"],
)
```

**编译与运行**:

```bash
# 编译
bazel build //:test_new_al_ilqr

# 运行
bazel run //:test_new_al_ilqr

# 或直接运行二进制
./bazel-bin/test_new_al_ilqr
```

### 5.3 pybind_extension - Python 扩展模块

**核心规则** (文件: `BUILD:91-104`)

```python
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

pybind_extension(
    name = "ilqr_pybind",             # 生成 ilqr_pybind.so
    srcs = ["ilqr_pybind.cc"],        # pybind11 绑定代码

    deps = [
        "//constraints:box_constraints",
        "//constraints:constraints_bind",
        "//model:node_bind",
        ":new_al_ilqr",
        "@eigen",
    ],

    copts = ["-O3", "-march=native", "-faligned-new", "-DEIGEN_VECTORIZE"],
)
```

#### pybind_extension 内部机制

`pybind_extension` 实际上是封装的 `cc_binary`，额外添加：

1. **自动链接 Python 库**: 添加 `@local_config_python//:python_headers` 依赖
2. **设置正确的链接标志**: `-shared -fPIC` 生成共享库
3. **命名约定**: 自动添加 `.so` 后缀 (Linux) 或 `.pyd` (Windows)
4. **Python 版本匹配**: 确保与 `python_configure()` 检测到的版本一致

**等价的 cc_binary 写法** (仅供理解):

```python
cc_binary(
    name = "ilqr_pybind.so",
    srcs = ["ilqr_pybind.cc"],
    deps = [
        # ... 应用依赖
        "@pybind11//:pybind11",
        "@local_config_python//:python_headers",
    ],
    linkshared = 1,  # 生成 .so
    linkstatic = 0,
)
```

### 5.4 py_library - Python 库包装

```python
py_library(
    name = "ilqr_pybind_lib",
    data = [":ilqr_pybind.so"],       # 将 .so 作为数据文件
    visibility = ["//visibility:public"],
)
```

**用途**: 允许其他 Bazel Python 目标依赖这个扩展模块 (本项目暂未使用)。

---

## 6. 编译优化配置

### 6.1 编译选项层次

本项目在多个层次配置编译选项：

```
1. .bazelrc (全局配置)
   build:opt --copt=-O3
        ↓
2. BUILD 文件 (目标级配置)
   cc_library(copts = ["-march=native"])
        ↓
3. 命令行覆盖 (临时配置)
   bazel build --copt=-g //:target
```

**优先级**: 命令行 > BUILD 文件 > .bazelrc

### 6.2 推荐的优化配置

**开发阶段** (快速迭代):

```bash
# .bazelrc
build:dev --compilation_mode=fastbuild  # 最小优化
build:dev --copt=-g                     # 包含调试符号
```

**性能测试** (最大性能):

```bash
# .bazelrc
build:perf --compilation_mode=opt
build:perf --copt=-O3
build:perf --copt=-march=native
build:perf --copt=-flto                 # 链接时优化
build:perf --linkopt=-flto
```

**生产发布** (兼容性 + 性能):

```bash
# .bazelrc
build:release --compilation_mode=opt
build:release --copt=-O3
build:release --copt=-march=x86-64      # 通用 x86-64 指令集 (不用 native)
build:release --strip=always            # 去除调试符号
```

### 6.3 Eigen 向量化优化

```python
# 所有使用 Eigen 的目标都应添加
copts = [
    "-DEIGEN_VECTORIZE",         # 启用向量化
    "-DEIGEN_VECTORIZE_SSE4_2",  # 显式指定 SSE4.2 (可选)
    "-DEIGEN_VECTORIZE_AVX2",    # 显式指定 AVX2 (需 CPU 支持)
]
```

**性能对比** (矩阵乘法 1000x1000):

| 配置 | 耗时 | 加速比 |
|------|------|-------|
| 无优化 (`-O0`) | 1200 ms | 1.0x |
| `-O3` | 300 ms | 4.0x |
| `-O3 -march=native -DEIGEN_VECTORIZE` | 80 ms | 15.0x |

---

## 7. 完整构建流程

### 7.1 冷启动构建 (首次)

```bash
cd cilqr/al_ilqr_cpp

# 步骤 1: 清理旧缓存 (可选)
bazel clean --expunge

# 步骤 2: 构建所有目标 (下载依赖 + 编译)
bazel build //...

# 实际执行流程:
# [1/50] Fetching @eigen from GitHub...
# [2/50] Fetching @pybind11...
# [3/50] Analyzing dependencies...
# [10/50] Compiling constraints/constraints.h
# [20/50] Compiling model/new_ilqr_node.h
# [40/50] Linking ilqr_pybind.so
# [50/50] Build completed
```

**时间估计**:
- 依赖下载: 30-60 秒 (取决于网络)
- 首次编译: 1-3 分钟 (取决于 CPU 核心数)

### 7.2 增量构建 (修改后)

**场景**: 修改了 `ilqr_pybind.cc` 文件

```bash
bazel build //:ilqr_pybind.so

# 实际执行流程:
# [1/3] Analyzing...
# [2/3] Compiling ilqr_pybind.cc       ← 只重新编译修改的文件
# [3/3] Linking ilqr_pybind.so
# Build completed in 5.2 seconds
```

**关键优势**: Bazel 通过 **文件内容哈希** 检测变化，只重新编译必要部分。

### 7.3 并行构建

```bash
# 使用所有 CPU 核心
bazel build --jobs=auto //...

# 限制并发任务数 (避免内存溢出)
bazel build --jobs=4 //...
```

### 7.4 跨目标构建

```bash
# 构建多个目标
bazel build //:ilqr_pybind.so //:test_new_al_ilqr //constraints:test_constraints

# 使用通配符 (构建所有测试)
bazel build //...:all

# 仅构建测试
bazel test //...
```

---

## 8. 常用命令与技巧

### 8.1 构建相关

```bash
# 构建单个目标
bazel build //:ilqr_pybind.so

# 构建所有目标
bazel build //...

# 清理构建产物 (保留缓存)
bazel clean

# 完全清理 (包括下载的依赖)
bazel clean --expunge

# 查看构建详细日志
bazel build //:ilqr_pybind.so --subcommands --verbose_failures

# 生成编译数据库 (用于 IDE)
bazel run @hedron_compile_commands//:refresh_all
```

### 8.2 查询与分析

```bash
# 查询所有目标
bazel query //...

# 查看目标依赖树
bazel query "deps(//:ilqr_pybind.so)" --output graph

# 查找依赖某个目标的所有目标
bazel query "rdeps(//..., @eigen//:eigen)"

# 查看为什么构建了某个目标
bazel query "somepath(//:ilqr_pybind.so, //model:new_ilqr_node)"

# 分析构建性能
bazel build //:ilqr_pybind.so --profile=profile.json
bazel analyze-profile profile.json
```

### 8.3 测试相关

```bash
# 运行所有测试
bazel test //...

# 运行特定测试
bazel test //:test_new_al_ilqr

# 显示测试输出
bazel test //:test_new_al_ilqr --test_output=all

# 调试失败的测试
bazel test //:test_new_al_ilqr --test_output=errors --verbose_failures
```

### 8.4 实用技巧

**① 使用本地镜像加速下载**

修改 `WORKSPACE`:

```python
http_archive(
    name = "eigen",
    urls = [
        "http://your-local-mirror/eigen-3.3.7.tar.gz",  # 本地镜像优先
        "https://github.com/eigenteam/eigen-git-mirror/archive/3.3.7.tar.gz",
    ],
)
```

**② 使用构建配置文件**

创建 `.bazelrc.user` (不提交到 Git):

```bash
build --python_path=/your/custom/python
build --config=opt
```

在 `.bazelrc` 中导入:

```bash
try-import %workspace%/.bazelrc.user
```

**③ 查看实际编译命令**

```bash
bazel build //:ilqr_pybind.so --subcommands | grep "ilqr_pybind.cc"
```

输出示例:

```bash
g++ -O3 -march=native -faligned-new -DEIGEN_VECTORIZE \
    -I external/eigen \
    -I external/pybind11/include \
    -fPIC -c ilqr_pybind.cc -o bazel-out/.../ilqr_pybind.pic.o
```

---

## 9. 问题排查与调试

### 9.1 依赖下载失败

**症状**:

```
ERROR: An error occurred during the fetch of repository 'eigen':
   java.io.IOException: Error downloading https://github.com/...
```

**解决方案**:

```bash
# 方案 1: 手动下载并放入缓存
mkdir -p ~/.cache/bazel/_bazel_$USER/cache/repos/v1/
wget https://github.com/eigenteam/eigen-git-mirror/archive/3.3.7.tar.gz \
     -O ~/.cache/bazel/_bazel_$USER/cache/repos/v1/3.3.7.tar.gz

# 方案 2: 使用代理
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port
bazel build //...

# 方案 3: 使用 distdir (预下载的归档)
bazel build --distdir=/path/to/downloaded/archives //...
```

### 9.2 Python 头文件找不到

**症状**:

```
fatal error: Python.h: No such file or directory
```

**解决方案**:

```bash
# Ubuntu/Debian
sudo apt install python3-dev

# 验证 Python 头文件位置
python3 -c "from sysconfig import get_paths; print(get_paths()['include'])"

# 强制 Bazel 重新配置 Python
bazel clean --expunge
bazel build //:ilqr_pybind.so
```

### 9.3 编译优化导致的错误

**症状**:

```
illegal instruction (core dumped)
```

**原因**: `-march=native` 在旧 CPU 上运行了新 CPU 编译的二进制

**解决方案**:

```bash
# 方案 1: 使用通用指令集
bazel build //:ilqr_pybind.so --copt=-march=x86-64

# 方案 2: 在目标 CPU 上重新编译
bazel clean
bazel build //:ilqr_pybind.so
```

### 9.4 符号未定义错误

**症状**:

```
undefined reference to `Eigen::internal::...`
```

**原因**: 依赖顺序错误或缺失

**调试步骤**:

```bash
# 1. 检查依赖图
bazel query "deps(//:ilqr_pybind.so)" --output graph | grep eigen

# 2. 检查链接顺序
bazel build //:ilqr_pybind.so --subcommands | grep "linkstatic"

# 3. 添加缺失的依赖
# 在 BUILD 文件中添加 "@eigen" 到 deps 列表
```

### 9.5 缓存问题

**症状**: 修改代码后构建无变化

```bash
# 清理缓存并重新构建
bazel clean
bazel build //:ilqr_pybind.so

# 如果还不行,完全清理
bazel clean --expunge
bazel shutdown
bazel build //:ilqr_pybind.so
```

### 9.6 查看详细错误信息

```bash
# 显示完整编译命令
bazel build //:ilqr_pybind.so --verbose_failures

# 显示所有子命令
bazel build //:ilqr_pybind.so --subcommands

# 生成详细日志
bazel build //:ilqr_pybind.so --explain=log.txt
cat log.txt
```

---

## 10. 高级主题

### 10.1 自定义编译工具链

创建 `.bazelrc`:

```bash
# 使用 clang 替代 gcc
build --action_env=CC=clang
build --action_env=CXX=clang++
```

### 10.2 交叉编译

```bash
# 编译 ARM64 版本 (需要交叉编译工具链)
bazel build //:ilqr_pybind.so \
    --cpu=aarch64 \
    --crosstool_top=@my_arm_toolchain//crosstool:toolchain
```

### 10.3 使用远程缓存 (加速 CI)

```bash
# .bazelrc
build --remote_cache=grpc://cache.example.com:9092
build --remote_upload_local_results=true
```

---

## 11. 总结

### 11.1 本项目 Bazel 使用要点

| 方面 | 技术点 |
|------|--------|
| **依赖管理** | `http_archive` 自动下载 Eigen/pybind11 |
| **Python 集成** | `python_configure()` 自动检测环境 |
| **编译优化** | `-O3 -march=native -DEIGEN_VECTORIZE` |
| **构建规则** | `cc_library`, `cc_binary`, `pybind_extension` |
| **增量构建** | 基于文件哈希的智能重新编译 |

### 11.2 最佳实践

1. **使用 .bazelrc 管理编译选项**: 避免在 BUILD 文件中硬编码
2. **合理拆分库**: 细粒度的 `cc_library` 提升增量构建效率
3. **显式声明依赖**: 避免隐式依赖导致的构建失败
4. **定期清理缓存**: `bazel clean` 解决大部分诡异问题
5. **使用查询工具**: `bazel query` 理解复杂依赖关系

### 11.3 进一步学习资源

- **官方文档**: https://bazel.build/
- **C++ 规则参考**: https://bazel.build/reference/be/c-cpp
- **pybind11_bazel**: https://github.com/pybind/pybind11_bazel
- **Bazel 最佳实践**: https://bazel.build/concepts/build-files

---

**文档版本**: v1.0
**最后更新**: 2025-10-19
**作者**: Claude Code
**适用 Bazel 版本**: 6.0+
