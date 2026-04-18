# Project3 矩阵乘法优化

12412639 王思宇

**目录**
[TOC]

## 1. 摘要

本项目主要使用了分块缓存、OpenMP多线程、SIMD、以及Out-of-Core等技术来优化矩阵乘法的性能，并与朴素矩阵乘法和OpenBLAS库的性能进行了对比。

## 2. 实验设计

### 2.1 实验环境

| 配置项 | 详细信息 |
| --- | --- |
| 设备机器 | 远程操控 Legion REN7000K-26IRX 台式电脑 |
| 处理器 | Intel(R) Core(TM) i7-14650HX (2.20 GHz) |
| 机带 RAM | 32.0 GB (31.6 GB 可用) |
| 系统类型 | 64 位操作系统, 基于 x64 的处理器 Windows 11 家庭中文版 |
| GCC | (x86_64-win32-seh-rev1, Built by MinGW-Builds project) 15.2.0 |

### 2.2 实验方法

终端打开指定的文件夹，输入
```shell
git clone https://github.com/feitianduowen/Proj3CPP.git
cd Proj3CPP
```

### 2.3 实验步骤
> 因为本实验焦点在于优化矩阵乘法的性能，所以本实验的所有矩阵乘法只考虑N*N的方阵乘法。

### 2.4 数据严谨性

时间测量代码方面使用了timespec_get函数获取高精度的现实世界时间戳，不受底层线程并行的影响。

实际测量时间=物理极限最快时间+系统噪音耗时(context switch、硬件中断等)。因此添加内循环和外循环来多次测量同一配置的性能，取最小值作为最终结果，以减小偶然因素的影响。内循环测量同一组参数多次，外循环则是重复整个测试过程多次。

现代CPU具有动态睿频(Turbo Boost)和温度墙限制，在连续进行多次大矩阵查时，CPU可能会因为过热而降低频率，导致后续测试的性能下降。为了减轻这个问题，我在每次测试之间加入了适当的休息时间（如10秒），以允许CPU降温并恢复正常频率。此外，在测试过程中，我监控了CPU温度，确保它不会过高。并且在每次运行代码前后间隔较长时间。

## 3. 优化过程解析

### 3.1 朴素矩阵乘法

### 3.2 SIMD 与 OpenMP

### 3.3

### 3.4 Out-of-Core

#### 3.4.1 N=64k 遇到的问题

如果仅仅使用`main.c`中的`matmul_imporved`或者`matmul_openblas`，在 N=65536 的规模下，一次矩阵乘法需要准备两个 struct Matrix (A 和 C，在本实验中为节省时间和内存开销，计算A*A)。

估算单纯内存消耗：

- 元素总数 = 64000 × 64000 = 4,096,000,000 个 float

- 每个单纯的 float 是 4 字节

- 单个矩阵大小：4,096,000,000 × 4 = 16,384,000,000 字节 ≈ 15.26 GB

因此，两个矩阵 A 和 C 的总内存需求约为 30.52 GB 的连续可用物理内存

因为电脑物理内存加虚拟系统分页没有远大于这个阈值（比如你只有 16GB 或 32GB RAM），操作系统会直接拒绝第二个矩阵的分配。

#### 3.4.2 选择OOC的原因

询问Gemini3.1针对上述问题给出了几个方案：

| 序号 | 方法                                                         | 缺点                                                         |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | 扩大虚拟内存。在 Windows 设置“高级系统设置” -> 性能“设置” -> “高级”选项卡 -> “虚拟内存” -> “更改”。取消自动管理，手动为SSD 硬盘分配一个大于 50000 MB（即 50GB）的虚拟内存大小。 | $O(N^3)$ 的非连续访存时间较长，预测需要几天甚至几月          |
| 2    | 把硬盘文件当成内存数组来读写。在 Linux 下使用 `mmap()`，在 Windows 下使用 `CreateFileMapping` 和 `MapViewOfFile`。 | 由于磁盘 I/O 速度比内存慢成千上万倍，遇到巨型矩阵计算依然会非常非常慢。 |
| 3    | 核外计算，具体内容如下3.4.4                                  | --                                                           |

综合考虑到时间和实现难度，最终选择了方案3：核外计算（OOC）。


#### 3.4.3 数学理论基础

设矩阵 $ A $ 大小为 $ m \times n $，矩阵 $ B $ 大小为 $ n \times p $，乘积 $ C = A \times B $ 大小为 $ m \times p $。

将 $ A, B, C $ 按下图方式划分为子块（各块内部的维度必须满足乘法规则）：

$$
A = \begin{bmatrix}
A_{11} & A_{12} & \cdots & A_{1k} \\
A_{21} & A_{22} & \cdots & A_{2k} \\
\vdots & \vdots & \ddots & \vdots \\
A_{t1} & A_{t2} & \cdots & A_{tk}
\end{bmatrix}, \quad
B = \begin{bmatrix}
B_{11} & B_{12} & \cdots & B_{1r} \\
B_{21} & B_{22} & \cdots & B_{2r} \\
\vdots & \vdots & \ddots & \vdots \\
B_{k1} & B_{k2} & \cdots & B_{kr}
\end{bmatrix}, \quad
C = \begin{bmatrix}
C_{11} & C_{12} & \cdots & C_{1r} \\
C_{21} & C_{22} & \cdots & C_{2r} \\
\vdots & \vdots & \ddots & \vdots \\
C_{t1} & C_{t2} & \cdots & C_{tr}
\end{bmatrix}
$$

那么分块乘法公式为：

$$
C_{ij} = \sum_{s=1}^{k} A_{is} \cdot B_{sj}
$$

其中每个 $ A_{is} \cdot B_{sj} $ 是子矩阵的乘积（普通矩阵乘法），结果的尺寸与 $ C_{ij} $ 一致。

**关键要求**：划分后，$ A $ 的列块数必须等于 $ B $ 的行块数（即 $ k $），且 $ A_{is} $ 的列数等于 $ B_{sj} $ 的行数。

**例子**：

设 $ A $ 为 $ 4 \times 4 $，$ B $ 为 $ 4 \times 4 $，块大小为 $ 2 \times 2 $。

$$
A = \begin{bmatrix}
a_{11} & a_{12} & a_{13} & a_{14} \\
a_{21} & a_{22} & a_{23} & a_{24} \\
a_{31} & a_{32} & a_{33} & a_{34} \\
a_{41} & a_{42} & a_{43} & a_{44}
\end{bmatrix}
= \begin{bmatrix}
A_{11} & A_{12} \\
A_{21} & A_{22}
\end{bmatrix},
\quad
B = \begin{bmatrix}
B_{11} & B_{12} \\
B_{21} & B_{22}
\end{bmatrix}
$$

则

$$
C = \begin{bmatrix}
A_{11}B_{11} + A_{12}B_{21} & A_{11}B_{12} + A_{12}B_{22} \\
A_{21}B_{11} + A_{22}B_{21} & A_{21}B_{12} + A_{22}B_{22}
\end{bmatrix}
$$

每个块乘法（如 $ A_{11}B_{11} $）是 $ 2 \times 2 $ 矩阵乘 $ 2 \times 2 $，结果也是 $ 2 \times 2 $。

#### 3.4.4 实现细节

OOC 的核心思想是将矩阵分块存储在磁盘上，每次只加载一个块到内存中进行计算，计算完成后再将结果写回磁盘。具体实现步骤如下：

#### 3.4.5 结果分析

因为从N=1024到65536的数据样本差了1e4数量级，如果直接使用耗时作为纵坐标绘图将无法清晰对比数据与观察趋势，所以我使用了对数坐标（log scale）和Normalized Time (time/$N^3$) 来绘制性能图表。通过对数和归一化坐标，我们可以更直观地观察不同块大小和矩阵规模下的性能差异。

<img src="..\important\results\ooc_barplot.png" alt="OOC Performance" style="zoom: 15%;" />

观察上图，我们可以得出以下结论：

- 不同N之间，随着矩阵规模的增加，总体耗时增加。
- 对于同一个N, 随着块大小的增大（同时块数量的减少），性能有显著提升。这是因为较大的块可以更有效地利用内存和磁盘的带宽，减少I/O操作的次数，从而提高整体计算效率。
- N越小的情况下，块大小对性能的影响越明显。这是因为较小的矩阵更容易被内存完全加载，较大的块可以更好地利用内存带宽。

#### 3.4.6 计算结果的准确性检验

为验证 OOC 实现的正确性，我选择了一个较小的矩阵规模（`N=1024,2048`），先用`main.c`中`create_matrix`和`randomize_matrix`建立随机数的矩阵A，然后将其存入磁盘。用`matmul_out_of_core`对磁盘文件`A_disk_test.bin`计算，结果存入`C_disk_test.bin`。把计算结果的磁盘文件读取为`struct Matrix`, 将其与`matmul_openblas`直接计算的结果，发现两者的结果在数值上非常接近，对于`N=1024`误差稳定在`2e-4`内, `N=2048`在`4e-4`内。这表明 OOC 实现能够正确地执行矩阵乘法，并且在数值精度方面没有显著问题。

```shell
Successfully read back C_disk.bin for verification.
  Max error: 1.831055e-04
Verification PASSED!Cleaning up temporary verification disk files...
Generating massive disk files...
checking Out-of-Core Computation...
Out-of-Core checking completed! 247755500 ns
Successfully read back C_disk.bin for verification.
  Max error: 3.662109e-04
Verification PASSED!Cleaning up temporary verification disk files...
N = 65536 , blocksize = 16384 ,...
```

## 4. 实验结果横向对比与分析

### 4.1 性能比较

### 4.2 结果分析

### 4.3 误差分析

## 5. 结论

### 5.1 主要发现

### 5.2 未来工作
