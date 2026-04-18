结合你的作业要求文档 (`q.md`) 来看，你提到的 "OpenMD" 应该是笔误，你指的应该是 **OpenMP** (Open Multi-Processing)。OpenMP 是一种用于共享内存并行系统的多线程程序设计方案，在深度学习和高性能计算中非常常用。

你目前的 `matmul_improved` 已经使用了 SIMD (AVX 指令集) 来向量化计算，这非常棒！接下来，我们可以引入 OpenMP 将单线程的 SIMD 计算扩展到多核多线程，进一步压榨 CPU 的性能。

### 1. 如何在现有代码中实现 OpenMP

在 C 语言中使用 OpenMP 非常简单，主要是通过编译器指令（Pragma）来告诉编译器如何并行化循环。你只需要在最外层循环上方加上一行 `#pragma omp parallel for` 即可。

修改后的 `matmul_improved` 代码如下：

```c
#include <omp.h> // 引入 OpenMP 头文件

int matmul_improved(int N, struct Matrix *A, struct Matrix *B, struct Matrix *C)
{
    if (!A || !B || !C || N <= 0 ||
        A->cols != N || A->rows != N ||
        B->cols != N || B->rows != N ||
        C->cols != N || C->rows != N ||
        !A->data || !B->data || !C->data)
    {
        return -1; // Invalid input
    }

    // 告诉编译器自动将最外层的 i 循环分配给多个 CPU 线程执行
    #pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j += 8)
        {
            __m256 sum = _mm256_setzero_ps();
            for (int k = 0; k < N; k++)
            {
                __m256 a = _mm256_set1_ps(A->data[i * N + k]);
                __m256 b = _mm256_loadu_ps(&B->data[k * N + j]);
                sum = _mm256_add_ps(sum, _mm256_mul_ps(a, b));
            }
            _mm256_storeu_ps(&C->data[i * N + j], sum);
        }
    }

    return 0;
}
```

**编译注意事项：**
使用 GCC 编译时，必须加上 `-fopenmp` 参数来开启 OpenMP 支持，同时保留优化和 AVX 支持参数：
```bash
gcc -O3 -mavx2 -mfma -fopenmp main.c test.c -o matmul_test -lopenblas -lm
```

---

### 2. 后续还可以干什么（进一步优化的方向）

你现在的版本（SIMD + OpenMP）比纯朴素版本会有巨大的提升，但距离 OpenBLAS 的极限性能还有一段路要走。为了在基准测试中获得更高分数，你可以尝试以下几个关键优化：

#### A. 修复当前代码的边界 Bug（紧急）
你目前的内层循环 `for (int j = 0; j < N; j += 8)` **假设了 $N$ 永远是 8 的倍数**。如果你的测试用例中有 $N=400$ 没问题，但如果是 $N=1000$（不是 8 的倍数），`j` 会越界访问内存，导致段错误或计算出脏数据。
* **解决方案**：在每次 `j` 循环结束或内部，添加一个标量处理（Scalar Fallback）的循环，用普通浮点乘法处理最后那几个不足 8 长度的尾部元素。

#### B. 缓存分块优化 (Cache Blocking / Tiling)
矩阵乘法是典型的**访存密集型**计算。当矩阵变得很大（比如 1Kx1K 甚至 8Kx8K）时，数据无法全部装入 CPU 的高速缓存（L1/L2 Cache）中。CPU 会花费大量时间在缓慢的内存读取上（Cache Miss）。
* **解决方案**：不要一次性算完一整行或一整列，而是将大矩阵切分成适合放入 L1/L2 Cache 的小块（例如 64x64 或 128x128 的 Block）。每次只对这些小块进行完整的乘法累加。这是 OpenBLAS 性能暴增的核心秘密。

#### C. 循环展开与寄存器分块 (Loop Unrolling & Register Blocking)
虽然你用了 AVX 计算单行的 8 个元素，但现代 CPU 有很多个 AVX 寄存器（通常有 16 个 YMM 寄存器）。
* **解决方案**：你可以同时计算 4 行和 4 列的结果（例如一个 4x4 的宏块），在最内层循环中一次性加载更多数据到多个寄存器中，隐藏 CPU 指令的延迟，减少从内存读取同一数据的次数。

#### D. 内存对齐 (Memory Alignment)
你目前使用的是 `_mm256_loadu_ps`（非对齐加载）。如果你的内存地址是 32 字节对齐的，可以使用 `_mm256_load_ps`（对齐加载），这在某些架构上速度会更快。
* **解决方案**：修改 `create_matrix` 中的 `malloc`，改为使用 `aligned_alloc(32, size)` 或 `posix_memalign`。

你打算先尝试修复 $N$ 不是 8 的倍数时的边界问题，还是想先直接编译跑一下现有的 AVX + OpenMP 代码看看提速效果？


> 上述代码中，有如下两个关键 `idea`，目的是减少单次实验的误差：
>
> - 本项目不使用 `time.h` 中的 `clock()`，而使用 `Linux` 系统库下的`time.h`。
>
>  这是由于后续使用`OpenMP`优化时，会使用多线程。`clock()`函数作为最常用的计时函数，返回`CPU`时钟计时单元数（clock tick）。只适用于单线程或单核心运行计时，但使用多线程运行时，`clock()`计时会引发问题，这是因为`clock()`在多线程返回的是`end-begin`的多个核心总共执行的时钟周期数，所以造成时间会偏大。
>
> - 对于较小矩阵进行矩阵乘法，连续进行多组相同计算，求其耗时的平均值。较小矩阵单次矩阵乘法所耗时间较小，难以直接测量，测量误差很大，连续进行多次计算求平均值，可显著降低误差。
> - 对于较大矩阵进行多次实验，求若干次实验中最小耗时，使得 `CPU` 能充分发挥峰值效能。

上述文本来自一位学长所写报告，下面的是我的代码的节选
```java
long long get_time_ns()
{
    struct timespec ts;
    // C11 标准时间函数，能获取纳秒精度（在 GCC/MinGW 下完美支持）
    timespec_get(&ts, TIME_UTC);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}
```
你觉得这个 `get_time_ns()` 函数的实现是否合理？它能否满足在多线程环境下准确测量时间的需求？相对于在Linux系统中使用 `time.h`，这个函数的性能和准确性如何？
"对于较大矩阵进行多次实验，求若干次实验中最小耗时"这句话是否合理，相对于求平均值，这种方法是否更能反映算法的性能？为什么？


如果仅仅使用`main.c`中的`matmul_imporved`或者`matmul_openblas`，在 N=65536 的规模下，一次矩阵乘法需要准备两个 struct Matrix (A 和 C，在本实验中为节省时间和内存开销，计算A*A)。
估算单纯内存消耗：

- 元素总数 = 64000 × 64000 = 4,096,000,000 个 float

- 每个单纯的 float 是 4 字节

- 单个矩阵大小：4,096,000,000 × 4 = 16,384,000,000 字节 ≈ 15.26 GB

因此，两个矩阵 A 和 C 的总内存需求约为 30.52 GB 的连续可用物理内存
因为电脑物理内存加虚拟系统分页没有远大于这个阈值（比如你只有 16GB 或 32GB RAM），操作系统会直接拒绝第二个矩阵的分配。

要列出“所有”关于矩阵乘法（GEMM）优化的论文是不太现实的。因为自计算机科学诞生以来，矩阵乘法一直被视为高性能计算（HPC）和深度学习的“圣杯”，相关的学术论文数以万计，涵盖了从纯数学理论复杂度到最底层的芯片指令集设计的方方面面。
不过，按照优化的演进方向，我为你梳理了 GEMM 发展史上最具里程碑意义、最值得精读的核心论文清单。你可以将它们作为一份“技能树”来探索：
1. 系统与硬件级访存优化（压榨算力极限）
在这个领域，研究的重点是如何通过缓存分块（Cache Blocking）、寄存器分配和 SIMD 向量化指令来打破“内存墙”（访存瓶颈）。
• Anatomy of High-Performance Matrix Multiplication (Kazushige Goto, Robert A. van de Geijn, 2008)
• 核心价值：这是必读中的必读。它是大名鼎鼎的 GotoBLAS（你测试过的 OpenBLAS 的前身）的核心理论。它极其深刻地剖析了如何根据 CPU 的 L1/L2 Cache 和 TLB（页表缓存）大小来精确设计矩阵的分块逻辑。读懂这篇，你就彻底明白了 BLAS 库为什么能跑满硬件的浮点峰值。
• Automated Empirical Optimization of Software and the ATLAS Project (R. Clint Whaley et al., 2001)
• 核心价值：提出了 ATLAS 库，开创了“自动调优（Autotuning）”的先河。它让程序在编译时自动运行大量测试，从而寻找当前硬件架构下最完美的循环展开数和分块大小。
2. 理论复杂度下限（数学算法突破）
这一分支主要由理论计算机科学家推动，目标是从代数结构上打破朴素算法 O(N^3) 的魔咒。
• Gaussian Elimination is not Optimal (Volker Strassen, 1969)
• 核心价值：Strassen 算法的开山之作，首次将渐进时间复杂度降到了 O(N^{2.807})。
• Matrix Multiplication via Arithmetic Progressions (Don Coppersmith, Shmuel Winograd, 1990)
• 核心价值：提出了著名的 Coppersmith-Winograd 算法，将复杂度降至 O(N^{2.3754})，并主导了该领域二十多年。
• New Bounds for Matrix Multiplication: from Alpha to Omega (Duan, Wu, Zhou, 2023)
• 核心价值：这是近年来的重大理论突破，由中国学者发表，将当前人类已知的渐进时间复杂度下限推进到了 O(N^{2.37188})。
3. 分布式与核外计算（大规模协同）
当问题规模超出单机内存（Out-of-Core）甚至单台服务器的算力时，研究重点转向如何在多级存储或网络中减少数据通信开销。
• SUMMA: Scalable Universal Matrix Multiplication Algorithm (Robert A. van de Geijn, Jerrell Watts, 1997)
• 核心价值：至今仍是基于 MPI（消息传递接口）的分布式集群中，最流行、最实用的矩阵乘法标准拓扑实现方案。
• Out-of-Core Algorithms for Sparse Matrix-Vector Multiplication (Toledo, 1999)
• 核心价值：系统性探讨了当矩阵大到无法装入主存时，如何利用磁盘 I/O 调度来实现高效的核外乘法计算。
4. 深度学习与 AI 驱动（新世代探索）
• Discovering faster matrix multiplication algorithms with reinforcement learning (DeepMind, Nature 2022)
• 核心价值：DeepMind 推出的 AlphaTensor，将寻找最优矩阵相乘方式转化为单人 3D 棋盘游戏，用强化学习发现了在特定矩阵规模下，比传统 Strassen 更快的局部算法方案。
如果你希望在代码实现层面寻求顿悟，我强烈建议先从 Anatomy of High-Performance Matrix Multiplication (2008) 这篇开始啃，它会让你对 Cache 行为产生极度清晰的物理直觉。
针对你接下来想要深挖的方向，你是想探究底层硬件指令集（比如 AVX 甚至国产架构体系的拓展指令）对软件生态的具体优化，还是想了解更宏观的大规模分布式系统调度方案？
