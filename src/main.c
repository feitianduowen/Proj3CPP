#include "main.h"

long long get_time_ns()
{
    struct timespec ts;
    // C11 标准时间函数，能获取纳秒精度（在 GCC/MinGW 下完美支持）
    timespec_get(&ts, TIME_UTC);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

void clear_matrix(Matrix const *mat)
{
    if (!mat || !mat->data) return;
    size_t count = mat->rows * mat->cols;
    for (size_t i = 0; i < count; i++)
        mat->data[i] = 0.0f;
}

int matmul_tp(size_t N, const Matrix *A, const Matrix *B, Matrix const *C){return -1;}

int matmul_plain(size_t N, const Matrix *A, const Matrix *B, Matrix const *C)
{
    // clear_matrix(C);

    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            float sum = 0.0f;
            for (size_t k = 0; k < N; k++)
            {
                sum += A->data[i * N + k] * B->data[k * N + j];
            }
            C->data[i * N + j] = sum;
        }
    }
    return 0;
}

int matmul_ikj(size_t N, const Matrix *A, const Matrix *B, Matrix const *C)
{
    // clear_matrix(C);
    for (size_t i = 0; i < N; i++)
    {
        for (size_t k = 0; k < N; k++)
        {
            for (size_t j = 0; j < N; j++)
            {
                C->data[i * N + j] += A->data[i * N + k] * B->data[k * N + j];
            }
        }
    }
    return 0;
}

// 辅助函数：带步长 (stride) 的矩阵加法
void strassen_add(size_t N, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            C[i * ldc + j] = A[i * lda + j] + B[i * ldb + j];
        }
    }
}

// 辅助函数：带步长 (stride) 的矩阵减法
void strassen_sub(size_t N, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            C[i * ldc + j] = A[i * lda + j] - B[i * ldb + j];
        }
    }
}

// 辅助函数：带步长的 ikj 矩阵乘法（适配 Strassen 的内存布局，无需拷贝子矩阵即可直接计算）
void matmul_ikj_stride(size_t N, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc) {
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            C[i * ldc + j] = 0.0f;
        }
        for (size_t k = 0; k < N; k++) {
            float a = A[i * lda + k];
            for (size_t j = 0; j < N; j++) {
                C[i * ldc + j] += a * B[k * ldb + j];
            }
        }
    }
}

// Strassen 递归核心计算逻辑
void strassen_recursive(size_t N, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc) {
    // 递归基：矩阵规模 <= 128 时，采用 ikj 算法处理以降低常数开销
    if (N <= 128) {
        matmul_ikj_stride(N, A, lda, B, ldb, C, ldc);
        return;
    }

    size_t half = N / 2;

    // 定位 4 个子矩阵的起始指针（不需要拷贝内存，只需正确偏移）
    const float* A11 = A;
    const float* A12 = A + half;
    const float* A21 = A + half * lda;
    const float* A22 = A + half * lda + half;

    const float* B11 = B;
    const float* B12 = B + half;
    const float* B21 = B + half * ldb;
    const float* B22 = B + half * ldb + half;

    float* C11 = C;
    float* C12 = C + half;
    float* C21 = C + half * ldc;
    float* C22 = C + half * ldc + half;

    float* buffer = (float*)malloc(9 * half * half * sizeof(float));
    if (!buffer) {
        fprintf(stderr, "Strassen memory allocation failed!\n");
        return;
    }

    float* P1 = buffer + 0 * half * half;
    float* P2 = buffer + 1 * half * half;
    float* P3 = buffer + 2 * half * half;
    float* P4 = buffer + 3 * half * half;
    float* P5 = buffer + 4 * half * half;
    float* P6 = buffer + 5 * half * half;
    float* P7 = buffer + 6 * half * half;
    float* T1 = buffer + 7 * half * half;
    float* T2 = buffer + 8 * half * half;

    // P1 = A11 * (B12 - B22)
    strassen_sub(half, B12, ldb, B22, ldb, T1, half);
    strassen_recursive(half, A11, lda, T1, half, P1, half);

    // P2 = (A11 + A12) * B22
    strassen_add(half, A11, lda, A12, lda, T1, half);
    strassen_recursive(half, T1, half, B22, ldb, P2, half);

    // P3 = (A21 + A22) * B11
    strassen_add(half, A21, lda, A22, lda, T1, half);
    strassen_recursive(half, T1, half, B11, ldb, P3, half);

    // P4 = A22 * (B21 - B11)
    strassen_sub(half, B21, ldb, B11, ldb, T1, half);
    strassen_recursive(half, A22, lda, T1, half, P4, half);

    // P5 = (A11 + A22) * (B11 + B22)
    strassen_add(half, A11, lda, A22, lda, T1, half);
    strassen_add(half, B11, ldb, B22, ldb, T2, half);
    strassen_recursive(half, T1, half, T2, half, P5, half);

    // P6 = (A12 - A22) * (B21 + B22)
    strassen_sub(half, A12, lda, A22, lda, T1, half);
    strassen_add(half, B21, ldb, B22, ldb, T2, half);
    strassen_recursive(half, T1, half, T2, half, P6, half);

    // P7 = (A11 - A21) * (B11 + B12)
    strassen_sub(half, A11, lda, A21, lda, T1, half);
    strassen_add(half, B11, ldb, B12, ldb, T2, half);
    strassen_recursive(half, T1, half, T2, half, P7, half);

    // 根据 P1~P7 组合求出目标子矩阵 C11, C12, C21, C22
    for (size_t i = 0; i < half; ++i) {
        for (size_t j = 0; j < half; ++j) {
            // C11 = P5 + P4 - P2 + P6
            C11[i * ldc + j] = P5[i * half + j] + P4[i * half + j] - P2[i * half + j] + P6[i * half + j];
            // C12 = P1 + P2
            C12[i * ldc + j] = P1[i * half + j] + P2[i * half + j];
            // C21 = P3 + P4
            C21[i * ldc + j] = P3[i * half + j] + P4[i * half + j];
            // C22 = P5 + P1 - P3 - P7
            C22[i * ldc + j] = P5[i * half + j] + P1[i * half + j] - P3[i * half + j] - P7[i * half + j];
        }
    }

    free(buffer);
}

// Strassen 顶层调用接口
int matmul_strassen(size_t N, const Matrix *A, const Matrix *B, Matrix const *C)
{
    strassen_recursive(N, A->data, N, B->data, N, C->data, N);
    return 0;
}

void improved_add(size_t N, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc) {
    for (size_t i = 0; i < N; ++i) {
        const float* pA = A + i * lda;
        const float* pB = B + i * ldb;
        float* pC = C + i * ldc;
        for (size_t j = 0; j < N; j += 32) {
            __m256 a0 = _mm256_loadu_ps(pA + j);
            __m256 b0 = _mm256_loadu_ps(pB + j);
            __m256 a1 = _mm256_loadu_ps(pA + j + 8);
            __m256 b1 = _mm256_loadu_ps(pB + j + 8);
            __m256 a2 = _mm256_loadu_ps(pA + j + 16);
            __m256 b2 = _mm256_loadu_ps(pB + j + 16);
            __m256 a3 = _mm256_loadu_ps(pA + j + 24);
            __m256 b3 = _mm256_loadu_ps(pB + j + 24);

            _mm256_storeu_ps(pC + j, _mm256_add_ps(a0, b0));
            _mm256_storeu_ps(pC + j + 8, _mm256_add_ps(a1, b1));
            _mm256_storeu_ps(pC + j + 16, _mm256_add_ps(a2, b2));
            _mm256_storeu_ps(pC + j + 24, _mm256_add_ps(a3, b3));
        }
    }
}

// 辅助函数：带步长 (stride) 的矩阵减法
void improved_sub(size_t N, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc) {
    for (size_t i = 0; i < N; ++i) {
        const float* pA = A + i * lda;
        const float* pB = B + i * ldb;
        float* pC = C + i * ldc;

        for (size_t j = 0; j < N; j += 32) {
            __m256 a0 = _mm256_loadu_ps(pA + j);
            __m256 b0 = _mm256_loadu_ps(pB + j);
            __m256 a1 = _mm256_loadu_ps(pA + j + 8);
            __m256 b1 = _mm256_loadu_ps(pB + j + 8);
            __m256 a2 = _mm256_loadu_ps(pA + j + 16);
            __m256 b2 = _mm256_loadu_ps(pB + j + 16);
            __m256 a3 = _mm256_loadu_ps(pA + j + 24);
            __m256 b3 = _mm256_loadu_ps(pB + j + 24);

            _mm256_storeu_ps(pC + j, _mm256_sub_ps(a0, b0));
            _mm256_storeu_ps(pC + j + 8, _mm256_sub_ps(a1, b1));
            _mm256_storeu_ps(pC + j + 16, _mm256_sub_ps(a2, b2));
            _mm256_storeu_ps(pC + j + 24, _mm256_sub_ps(a3, b3));
        }
    }
}

// 辅助函数：带步长的 ikj 矩阵乘法（适配 Strassen 的内存布局，无需拷贝子矩阵即可直接计算）
void improved_stride(size_t N, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc) {
    for (size_t i = 0; i < N; i += 4) {
        for (size_t k = 0; k < N; k++) {
            __m256 va0 = _mm256_set1_ps(A[(i + 0) * lda + k]);
            __m256 va1 = _mm256_set1_ps(A[(i + 1) * lda + k]);
            __m256 va2 = _mm256_set1_ps(A[(i + 2) * lda + k]);
            __m256 va3 = _mm256_set1_ps(A[(i + 3) * lda + k]);

            const float* pB = B + k * ldb;
            float* pC0 = C + (i + 0) * ldc;
            float* pC1 = C + (i + 1) * ldc;
            float* pC2 = C + (i + 2) * ldc;
            float* pC3 = C + (i + 3) * ldc;

            for (size_t j = 0; j < N; j += 16) {
                __m256 vb0 = _mm256_loadu_ps(pB + j);
                __m256 vb1 = _mm256_loadu_ps(pB + j + 8);

                _mm256_storeu_ps(pC0 + j, _mm256_add_ps(_mm256_loadu_ps(pC0 + j), _mm256_mul_ps(va0, vb0)));
                _mm256_storeu_ps(pC0 + j + 8, _mm256_add_ps(_mm256_loadu_ps(pC0 + j + 8), _mm256_mul_ps(va0, vb1)));

                _mm256_storeu_ps(pC1 + j, _mm256_add_ps(_mm256_loadu_ps(pC1 + j), _mm256_mul_ps(va1, vb0)));
                _mm256_storeu_ps(pC1 + j + 8, _mm256_add_ps(_mm256_loadu_ps(pC1 + j + 8), _mm256_mul_ps(va1, vb1)));

                _mm256_storeu_ps(pC2 + j, _mm256_add_ps(_mm256_loadu_ps(pC2 + j), _mm256_mul_ps(va2, vb0)));
                _mm256_storeu_ps(pC2 + j + 8, _mm256_add_ps(_mm256_loadu_ps(pC2 + j + 8), _mm256_mul_ps(va2, vb1)));

                _mm256_storeu_ps(pC3 + j, _mm256_add_ps(_mm256_loadu_ps(pC3 + j), _mm256_mul_ps(va3, vb0)));
                _mm256_storeu_ps(pC3 + j + 8, _mm256_add_ps(_mm256_loadu_ps(pC3 + j + 8), _mm256_mul_ps(va3, vb1)));
            }
        }
    }
}

// Strassen 递归核心计算逻辑
void improved_recursive(size_t N, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc) {

    if (N <= 128) {
        improved_stride(N, A, lda, B, ldb, C, ldc);
        return;
    }

    size_t half = N / 2;

    // 定位 4 个子矩阵的起始指针（不需要拷贝内存，只需正确偏移）
    const float* A11 = A;
    const float* A12 = A + half;
    const float* A21 = A + half * lda;
    const float* A22 = A21 + half;

    const float* B11 = B;
    const float* B12 = B + half;
    const float* B21 = B + half * ldb;
    const float* B22 = B21 + half;

    float* C11 = C;
    float* C12 = C + half;
    float* C21 = C + half * ldc;
    float* C22 = C21 + half;

    // 优化：分配 17 个区块，确保7个任务有独立的临时缓存（T区），为后续多核并行扫清数据竞争隐患
    float* buffer = (float*)malloc(17 * half * half * sizeof(float));
    if (!buffer) {
        fprintf(stderr, "Strassen memory allocation failed!\n");
        return;
    }
    size_t h2= half * half;
    float* P1 = buffer;
    float* P2 = buffer + 1 * h2;
    float* P3 = buffer + 2 * h2;
    float* P4 = buffer + 3 * h2;
    float* P5 = buffer + 4 * h2;
    float* P6 = buffer + 5 * h2;
    float* P7 = buffer + 6 * h2;

    // 给 7 个独立任务各自配备专属的 Temp 缓冲区 (10个块)
    float* TP1 = buffer + 7 * h2;
    float* TP2 = buffer + 8 * h2;
    float* TP3 = buffer + 9 * h2;
    float* TP4 = buffer + 10 * h2;
    float* TP5_1 = buffer + 11 * h2;
    float* TP5_2 = buffer + 12 * h2;
    float* TP6_1 = buffer + 13 * h2;
    float* TP6_2 = buffer + 14 * h2;
    float* TP7_1 = buffer + 15 * h2;
    float* TP7_2 = buffer + 16 * h2;

    // 优化：使用 OpenMP Task 将互不相干的 P1~P7 扔给多核线程池并发计算
    #pragma omp task default(none) shared(half,lda,ldb,P1,TP1,A11,B12,B22)
    {
        // P1 = A11 * (B12 - B22)
        improved_sub(half, B12, ldb, B22, ldb, TP1, half);
        improved_recursive(half, A11, lda, TP1, half, P1, half);
    }

    #pragma omp task default(none) shared(half,lda,ldb,P2,TP2,A11,A12,B22)
    {
        // P2 = (A11 + A12) * B22
        improved_add(half, A11, lda, A12, lda, TP2, half);
        improved_recursive(half, TP2, half, B22, ldb, P2, half);
    }

    #pragma omp task default(none) shared(half,lda,ldb,P3,TP3,A21,A22,B11)
    {
        // P3 = (A21 + A22) * B11
        improved_add(half, A21, lda, A22, lda, TP3, half);
        improved_recursive(half, TP3, half, B11, ldb, P3, half);
    }

    #pragma omp task default(none) shared(half,lda,ldb,P4,TP4,A22,B11,B21)
    {
        // P4 = A22 * (B21 - B11)
        improved_sub(half, B21, ldb, B11, ldb, TP4, half);
        improved_recursive(half, A22, lda, TP4, half, P4, half);
    }

    #pragma omp task default(none) shared(half,lda,ldb,P5,TP5_1,TP5_2,A11,A22,B11,B22)
    {
        // P5 = (A11 + A22) * (B11 + B22)
        improved_add(half, A11, lda, A22, lda, TP5_1, half);
        improved_add(half, B11, ldb, B22, ldb, TP5_2, half);
        improved_recursive(half, TP5_1, half, TP5_2, half, P5, half);
    }

    #pragma omp task default(none) shared(half,lda,ldb,P6,TP6_1,TP6_2,A12,A22,B21,B22)
    {
        // P6 = (A12 - A22) * (B21 + B22)
        improved_sub(half, A12, lda, A22, lda, TP6_1, half);
        improved_add(half, B21, ldb, B22, ldb, TP6_2, half);
        improved_recursive(half, TP6_1, half, TP6_2, half, P6, half);
    }

    #pragma omp task default(none) shared(half,lda,ldb,P7,TP7_1,TP7_2,A11,A21,B11,B12)
    {
        // P7 = (A11 - A21) * (B11 + B12)
        improved_sub(half, A11, lda, A21, lda, TP7_1, half);
        improved_add(half, B11, ldb, B12, ldb, TP7_2, half);
        improved_recursive(half, TP7_1, half, TP7_2, half, P7, half);
    }

    // 阻塞并等待这 7 个任务全部执行完毕，再合并最终结果
    // #pragma omp taskwait
#pragma omp parallel for collapse(2) default(none) \
shared(half, P1, P2, P3, P4, P5, P6, P7, C11, C12, C21, C22, ldc) \
// private(i, j)
    // 根据 P1~P7 组合求出目标子矩阵 C11, C12, C21, C22
    for (size_t i = 0; i < half; ++i) {
        size_t ih=i*half;
        const float* p1 = P1 + ih;
        const float* p2 = P2 + ih;
        const float* p3 = P3 + ih;
        const float* p4 = P4 + ih;
        const float* p5 = P5 + ih;
        const float* p6 = P6 + ih;
        const float* p7 = P7 + ih;

        float* c11 = C11 + i * ldc;
        float* c12 = C12 + i * ldc;
        float* c21 = C21 + i * ldc;
        float* c22 = C22 + i * ldc;


        // AVX 向量化，每次处理 32 个 float (循环展开)
        for (size_t j = 0; j < half; j += 32) {
            __m256 v1_0 = _mm256_loadu_ps(p1 + j);
            __m256 v2_0 = _mm256_loadu_ps(p2 + j);
            __m256 v3_0 = _mm256_loadu_ps(p3 + j);
            __m256 v4_0 = _mm256_loadu_ps(p4 + j);
            __m256 v5_0 = _mm256_loadu_ps(p5 + j);
            __m256 v6_0 = _mm256_loadu_ps(p6 + j);
            __m256 v7_0 = _mm256_loadu_ps(p7 + j);

            __m256 v1_1 = _mm256_loadu_ps(p1 + j + 8);
            __m256 v2_1 = _mm256_loadu_ps(p2 + j + 8);
            __m256 v3_1 = _mm256_loadu_ps(p3 + j + 8);
            __m256 v4_1 = _mm256_loadu_ps(p4 + j + 8);
            __m256 v5_1 = _mm256_loadu_ps(p5 + j + 8);
            __m256 v6_1 = _mm256_loadu_ps(p6 + j + 8);
            __m256 v7_1 = _mm256_loadu_ps(p7 + j + 8);

            __m256 v1_2 = _mm256_loadu_ps(p1 + j + 16);
            __m256 v2_2 = _mm256_loadu_ps(p2 + j + 16);
            __m256 v3_2 = _mm256_loadu_ps(p3 + j + 16);
            __m256 v4_2 = _mm256_loadu_ps(p4 + j + 16);
            __m256 v5_2 = _mm256_loadu_ps(p5 + j + 16);
            __m256 v6_2 = _mm256_loadu_ps(p6 + j + 16);
            __m256 v7_2 = _mm256_loadu_ps(p7 + j + 16);

            __m256 v1_3 = _mm256_loadu_ps(p1 + j + 24);
            __m256 v2_3 = _mm256_loadu_ps(p2 + j + 24);
            __m256 v3_3 = _mm256_loadu_ps(p3 + j + 24);
            __m256 v4_3 = _mm256_loadu_ps(p4 + j + 24);
            __m256 v5_3 = _mm256_loadu_ps(p5 + j + 24);
            __m256 v6_3 = _mm256_loadu_ps(p6 + j + 24);
            __m256 v7_3 = _mm256_loadu_ps(p7 + j + 24);

            // C11 = P5 + P4 - P2 + P6
            _mm256_storeu_ps(c11 + j, _mm256_add_ps(_mm256_sub_ps(_mm256_add_ps(v5_0, v4_0), v2_0), v6_0));
            _mm256_storeu_ps(c11 + j + 8, _mm256_add_ps(_mm256_sub_ps(_mm256_add_ps(v5_1, v4_1), v2_1), v6_1));
            _mm256_storeu_ps(c11 + j + 16, _mm256_add_ps(_mm256_sub_ps(_mm256_add_ps(v5_2, v4_2), v2_2), v6_2));
            _mm256_storeu_ps(c11 + j + 24, _mm256_add_ps(_mm256_sub_ps(_mm256_add_ps(v5_3, v4_3), v2_3), v6_3));

            // C12 = P1 + P2
            _mm256_storeu_ps(c12 + j, _mm256_add_ps(v1_0, v2_0));
            _mm256_storeu_ps(c12 + j + 8, _mm256_add_ps(v1_1, v2_1));
            _mm256_storeu_ps(c12 + j + 16, _mm256_add_ps(v1_2, v2_2));
            _mm256_storeu_ps(c12 + j + 24, _mm256_add_ps(v1_3, v2_3));

            // C21 = P3 + P4
            _mm256_storeu_ps(c21 + j, _mm256_add_ps(v3_0, v4_0));
            _mm256_storeu_ps(c21 + j + 8, _mm256_add_ps(v3_1, v4_1));
            _mm256_storeu_ps(c21 + j + 16, _mm256_add_ps(v3_2, v4_2));
            _mm256_storeu_ps(c21 + j + 24, _mm256_add_ps(v3_3, v4_3));

            // C22 = P5 + P1 - P3 - P7
            _mm256_storeu_ps(c22 + j, _mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(v5_0, v1_0), v3_0), v7_0));
            _mm256_storeu_ps(c22 + j + 8, _mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(v5_1, v1_1), v3_1), v7_1));
            _mm256_storeu_ps(c22 + j + 16, _mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(v5_2, v1_2), v3_2), v7_2));
            _mm256_storeu_ps(c22 + j + 24, _mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(v5_3, v1_3), v3_3), v7_3));
        }

    }

    free(buffer);
}

// Strassen 顶层调用接口
int matmul_improved(size_t N, const Matrix *A, const Matrix *B, Matrix const *C)
{
    improved_recursive(N, A->data, N, B->data, N, C->data, N);
    return 0;
}



void improved3_leaf(size_t N, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc) {
    for (size_t i = 0; i < N; ++i) {
        memset(&C[i * ldc], 0, N * sizeof(float));
    }

    const int BLOCK = 128;
    for (size_t i_b = 0; i_b < N; i_b += BLOCK) {
        for (size_t j_b = 0; j_b < N; j_b += BLOCK) {
            for (size_t k_b = 0; k_b < N; k_b += BLOCK) {
                size_t i_max = (i_b + BLOCK < N) ? i_b + BLOCK : N;
                size_t j_max = (j_b + BLOCK < N) ? j_b + BLOCK : N;
                size_t k_max = (k_b + BLOCK < N) ? k_b + BLOCK : N;

                for (size_t i = i_b; i < i_max; i += 4) {
                    for (size_t j = j_b; j < j_max; j += 8) {
                        __m256 c0 = _mm256_loadu_ps(&C[(i + 0) * ldc + j]);
                        __m256 c1 = _mm256_loadu_ps(&C[(i + 1) * ldc + j]);
                        __m256 c2 = _mm256_loadu_ps(&C[(i + 2) * ldc + j]);
                        __m256 c3 = _mm256_loadu_ps(&C[(i + 3) * ldc + j]);

                        for (size_t k = k_b; k < k_max; k++) {
                            __m256 b = _mm256_loadu_ps(&B[k * ldb + j]);
                            c0 = _mm256_add_ps(c0, _mm256_mul_ps(_mm256_set1_ps(A[(i + 0) * lda + k]), b));
                            c1 = _mm256_add_ps(c1, _mm256_mul_ps(_mm256_set1_ps(A[(i + 1) * lda + k]), b));
                            c2 = _mm256_add_ps(c2, _mm256_mul_ps(_mm256_set1_ps(A[(i + 2) * lda + k]), b));
                            c3 = _mm256_add_ps(c3, _mm256_mul_ps(_mm256_set1_ps(A[(i + 3) * lda + k]), b));
                        }

                        _mm256_storeu_ps(&C[(i + 0) * ldc + j], c0);
                        _mm256_storeu_ps(&C[(i + 1) * ldc + j], c1);
                        _mm256_storeu_ps(&C[(i + 2) * ldc + j], c2);
                        _mm256_storeu_ps(&C[(i + 3) * ldc + j], c3);
                    }
                }
            }
        }
    }
}
void improved3_add(size_t N, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; j += 8) {
            _mm256_storeu_ps(&C[i * ldc + j],
                _mm256_add_ps(_mm256_loadu_ps(&A[i * lda + j]), _mm256_loadu_ps(&B[i * ldb + j])));
        }
    }
}

void improved3_sub(size_t N, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; j += 8) {
            _mm256_storeu_ps(&C[i * ldc + j],
                _mm256_sub_ps(_mm256_loadu_ps(&A[i * lda + j]), _mm256_loadu_ps(&B[i * ldb + j])));
        }
    }
}

// -------------------------------------------------------------------------
// 3. 纯单线程顺序递归核心 (严格共享预分配的 Buffer)
// -------------------------------------------------------------------------
void improved3_seq(size_t N, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc, float* buffer) {
    if (N <= 256) {
        improved3_leaf(N, A, lda, B, ldb, C, ldc);
        return;
    }

    size_t half = N / 2;
    size_t h2 = half * half;

    float *P1 = buffer, *P2 = buffer + h2, *P3 = buffer + 2*h2, *P4 = buffer + 3*h2;
    float *P5 = buffer + 4*h2, *P6 = buffer + 5*h2, *P7 = buffer + 6*h2;
    float *T1 = buffer + 7*h2, *T2 = buffer + 8*h2, *T3 = buffer + 9*h2, *T4 = buffer + 10*h2;
    float *T5_1 = buffer + 11*h2, *T5_2 = buffer + 12*h2;
    float *T6_1 = buffer + 13*h2, *T6_2 = buffer + 14*h2;
    float *T7_1 = buffer + 15*h2, *T7_2 = buffer + 16*h2;

    // 把剩余空间交给下一层递归，避免任何 malloc
    float *next_buffer = buffer + 17 * h2;

    const float *A11 = A, *A12 = A + half, *A21 = A + half * lda, *A22 = A21 + half;
    const float *B11 = B, *B12 = B + half, *B21 = B + half * ldb, *B22 = B21 + half;

    improved3_sub(half, B12, ldb, B22, ldb, T1, half);
    improved3_seq(half, A11, lda, T1, half, P1, half, next_buffer);

    improved3_add(half, A11, lda, A12, lda, T2, half);
    improved3_seq(half, T2, half, B22, ldb, P2, half, next_buffer);

    improved3_add(half, A21, lda, A22, lda, T3, half);
    improved3_seq(half, T3, half, B11, ldb, P3, half, next_buffer);

    improved3_sub(half, B21, ldb, B11, ldb, T4, half);
    improved3_seq(half, A22, lda, T4, half, P4, half, next_buffer);

    improved3_add(half, A11, lda, A22, lda, T5_1, half);
    improved3_add(half, B11, ldb, B22, ldb, T5_2, half);
    improved3_seq(half, T5_1, half, T5_2, half, P5, half, next_buffer);

    improved3_sub(half, A12, lda, A22, lda, T6_1, half);
    improved3_add(half, B21, ldb, B22, ldb, T6_2, half);
    improved3_seq(half, T6_1, half, T6_2, half, P6, half, next_buffer);

    improved3_sub(half, A11, lda, A21, lda, T7_1, half);
    improved3_add(half, B11, ldb, B12, ldb, T7_2, half);
    improved3_seq(half, T7_1, half, T7_2, half, P7, half, next_buffer);

    float *C11 = C, *C12 = C + half, *C21 = C + half * ldc, *C22 = C21 + half;
    for (size_t i = 0; i < half; i++) {
        for (size_t j = 0; j < half; j += 8) {
            __m256 p1 = _mm256_loadu_ps(P1 + i * half + j);
            __m256 p2 = _mm256_loadu_ps(P2 + i * half + j);
            __m256 p3 = _mm256_loadu_ps(P3 + i * half + j);
            __m256 p4 = _mm256_loadu_ps(P4 + i * half + j);
            __m256 p5 = _mm256_loadu_ps(P5 + i * half + j);
            __m256 p6 = _mm256_loadu_ps(P6 + i * half + j);
            __m256 p7 = _mm256_loadu_ps(P7 + i * half + j);

            _mm256_storeu_ps(C11 + i * ldc + j, _mm256_add_ps(_mm256_sub_ps(_mm256_add_ps(p5, p4), p2), p6));
            _mm256_storeu_ps(C12 + i * ldc + j, _mm256_add_ps(p1, p2));
            _mm256_storeu_ps(C21 + i * ldc + j, _mm256_add_ps(p3, p4));
            _mm256_storeu_ps(C22 + i * ldc + j, _mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(p5, p1), p3), p7));
        }
    }
}

int matmul_improved3(size_t N, const Matrix *A, const Matrix *B, Matrix const *C)
{
    if (N <= 256 || (N % 256 != 0)) {
        improved3_leaf(N, A->data, N, B->data, N, C->data, N);
        return 0;
    }

    size_t half = N / 2;
    size_t h2 = half * half;

    // 严密计算所需的空间：
    // 第一层自身需要 P[7] 和 T[10]，共 17 * h2
    // 每个子任务(共7个)向下递归的深度缓冲总和是收敛的等比数列，严格小于 6 * h2
    // 总分配量 = 17 * h2 + 7 * 6 * h2 = 59 * h2
    // 采用普通 malloc，完全规避 _aligned
    float* buffer_top = (float*)malloc(59 * h2 * sizeof(float));
    if (!buffer_top) {
        printf("Improved1 memory allocation failed!\n");
        return -1;
    }

    float *P[7];
    for (int i=0; i<7; i++) P[i] = buffer_top + i * h2;
    float *T[10];
    for (int i=0; i<10; i++) T[i] = buffer_top + (7 + i) * h2;

    const float *A11 = A->data, *A12 = A->data + half, *A21 = A->data + half * N, *A22 = A21 + half;
    const float *B11 = B->data, *B12 = B->data + half, *B21 = B->data + half * N, *B22 = B21 + half;

    #pragma omp parallel default(none) shared(half, N, A11, A12, A21, A22, B11, B12, B21, B22, P, T, buffer_top, h2)
    #pragma omp single
    {
        // 向每个 task 中传递它专属的预分配空间区域，杜绝所有函数内的 malloc/free 开销
        #pragma omp task
        {
            float* task_buf = buffer_top + 17 * h2 + 0 * 6 * h2;
            improved3_sub(half, B12, N, B22, N, T[0], half);
            improved3_seq(half, A11, N, T[0], half, P[0], half, task_buf);
        }
        #pragma omp task
        {
            float* task_buf = buffer_top + 17 * h2 + 1 * 6 * h2;
            improved3_add(half, A11, N, A12, N, T[1], half);
            improved3_seq(half, T[1], half, B22, N, P[1], half, task_buf);
        }
        #pragma omp task
        {
            float* task_buf = buffer_top + 17 * h2 + 2 * 6 * h2;
            improved3_add(half, A21, N, A22, N, T[2], half);
            improved3_seq(half, T[2], half, B11, N, P[2], half, task_buf);
        }
        #pragma omp task
        {
            float* task_buf = buffer_top + 17 * h2 + 3 * 6 * h2;
            improved3_sub(half, B21, N, B11, N, T[3], half);
            improved3_seq(half, A22, N, T[3], half, P[3], half, task_buf);
        }
        #pragma omp task
        {
            float* task_buf = buffer_top + 17 * h2 + 4 * 6 * h2;
            improved3_add(half, A11, N, A22, N, T[4], half);
            improved3_add(half, B11, N, B22, N, T[5], half);
            improved3_seq(half, T[4], half, T[5], half, P[4], half, task_buf);
        }
        #pragma omp task
        {
            float* task_buf = buffer_top + 17 * h2 + 5 * 6 * h2;
            improved3_sub(half, A12, N, A22, N, T[6], half);
            improved3_add(half, B21, N, B22, N, T[7], half);
            improved3_seq(half, T[6], half, T[7], half, P[5], half, task_buf);
        }
        #pragma omp task
        {
            float* task_buf = buffer_top + 17 * h2 + 6 * 6 * h2;
            improved3_sub(half, A11, N, A21, N, T[8], half);
            improved3_add(half, B11, N, B12, N, T[9], half);
            improved3_seq(half, T[8], half, T[9], half, P[6], half, task_buf);
        }
    }

    float *C11 = C->data, *C12 = C->data + half, *C21 = C->data + half * N, *C22 = C->data + half * N + half;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < half; i++) {
        for (size_t j = 0; j < half; j += 8) {
            __m256 p1 = _mm256_loadu_ps(P[0] + i * half + j);
            __m256 p2 = _mm256_loadu_ps(P[1] + i * half + j);
            __m256 p3 = _mm256_loadu_ps(P[2] + i * half + j);
            __m256 p4 = _mm256_loadu_ps(P[3] + i * half + j);
            __m256 p5 = _mm256_loadu_ps(P[4] + i * half + j);
            __m256 p6 = _mm256_loadu_ps(P[5] + i * half + j);
            __m256 p7 = _mm256_loadu_ps(P[6] + i * half + j);

            _mm256_storeu_ps(C11 + i * N + j, _mm256_add_ps(_mm256_sub_ps(_mm256_add_ps(p5, p4), p2), p6));
            _mm256_storeu_ps(C12 + i * N + j, _mm256_add_ps(p1, p2));
            _mm256_storeu_ps(C21 + i * N + j, _mm256_add_ps(p3, p4));
            _mm256_storeu_ps(C22 + i * N + j, _mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(p5, p1), p3), p7));
        }
    }

    free(buffer_top);
    return 0;
}




// -------------------------------------------------------------------------
// 2. AVX 辅助加减法 (步长为 8，确保所有尺寸安全兼容)
// -------------------------------------------------------------------------

int matmul_aligned(size_t N, const Matrix *A, const Matrix *B, Matrix const *C)
{
    if (N <= 128)
    {
        for (size_t i = 0; i < N; i += 4)
        {
            for (size_t j = 0; j < N; j += 8)
            {
                __m256 c0 = _mm256_load_ps(&C->data[(i + 0) * N + j]);
                __m256 c1 = _mm256_load_ps(&C->data[(i + 1) * N + j]);
                __m256 c2 = _mm256_load_ps(&C->data[(i + 2) * N + j]);
                __m256 c3 = _mm256_load_ps(&C->data[(i + 3) * N + j]);

                for (size_t k = 0; k < N; k++)
                {
                    __m256 b = _mm256_load_ps(&B->data[k * N + j]);
                    c0 = _mm256_add_ps(c0, _mm256_mul_ps(_mm256_set1_ps(A->data[(i + 0) * N + k]), b));
                    c1 = _mm256_add_ps(c1, _mm256_mul_ps(_mm256_set1_ps(A->data[(i + 1) * N + k]), b));
                    c2 = _mm256_add_ps(c2, _mm256_mul_ps(_mm256_set1_ps(A->data[(i + 2) * N + k]), b));
                    c3 = _mm256_add_ps(c3, _mm256_mul_ps(_mm256_set1_ps(A->data[(i + 3) * N + k]), b));
                }

                _mm256_store_ps(&C->data[(i + 0) * N + j], c0);
                _mm256_store_ps(&C->data[(i + 1) * N + j], c1);
                _mm256_store_ps(&C->data[(i + 2) * N + j], c2);
                _mm256_store_ps(&C->data[(i + 3) * N + j], c3);
            }
        }
        return 0;
    }

    const int BLOCK = 128;

#pragma omp parallel for collapse(2) default(none) shared(N, A, B, C,BLOCK)
    for (size_t i_b = 0; i_b < N; i_b += BLOCK)
    {
        for (size_t j_b = 0; j_b < N; j_b += BLOCK)
        {
            for (size_t k_b = 0; k_b < N; k_b += BLOCK)
            {
                size_t i_max = (i_b + BLOCK < N) ? i_b + BLOCK : N;
                size_t j_max = (j_b + BLOCK < N) ? j_b + BLOCK : N;
                size_t k_max = (k_b + BLOCK < N) ? k_b + BLOCK : N;

                for (size_t i = i_b; i < i_max; i += 4)
                {
                    for (size_t j = j_b; j < j_max; j += 8)
                    {
                        __m256 c0 = _mm256_load_ps(&C->data[(i + 0) * N + j]);
                        __m256 c1 = _mm256_load_ps(&C->data[(i + 1) * N + j]);
                        __m256 c2 = _mm256_load_ps(&C->data[(i + 2) * N + j]);
                        __m256 c3 = _mm256_load_ps(&C->data[(i + 3) * N + j]);

                        for (size_t k = k_b; k < k_max; k++)
                        {
                            __m256 b = _mm256_load_ps(&B->data[k * N + j]);
                            c0 = _mm256_add_ps(c0, _mm256_mul_ps(_mm256_set1_ps(A->data[(i + 0) * N + k]), b));
                            c1 = _mm256_add_ps(c1, _mm256_mul_ps(_mm256_set1_ps(A->data[(i + 1) * N + k]), b));
                            c2 = _mm256_add_ps(c2, _mm256_mul_ps(_mm256_set1_ps(A->data[(i + 2) * N + k]), b));
                            c3 = _mm256_add_ps(c3, _mm256_mul_ps(_mm256_set1_ps(A->data[(i + 3) * N + k]), b));
                        }

                        _mm256_store_ps(&C->data[(i + 0) * N + j], c0);
                        _mm256_store_ps(&C->data[(i + 1) * N + j], c1);
                        _mm256_store_ps(&C->data[(i + 2) * N + j], c2);
                        _mm256_store_ps(&C->data[(i + 3) * N + j], c3);
                    }
                }
            }
        }
    }

    return 0;
}

int matmul_openblas(size_t N, const Matrix *A, const Matrix *B, Matrix const *C)
{
    if (!A || !B || !C || N <= 0)
        return -1;
    int N1= (int)N;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N1, N1, N1,
                1.0f, A->data, N1,
                B->data, N1,
                0.0f, C->data, N1);
    return 0;
}

int matmul_openblas_accumulate(size_t N, const Matrix *A, const Matrix *B, Matrix const *C)
{
    int N1= (int)N;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N1, N1, N1,
                1.0f, A->data, N1,
                B->data, N1,
                1.0f, C->data, N1); // <--- 这里改成 1.0f 就自带累加了！
    return 0;
}

int matmul_improved2(size_t N, const Matrix *A, const Matrix *B, Matrix const *C){
    // printf("Running improved2 matmul with N=%zu\n", N);


    // 针对小矩阵 (N <= 128) 的特化全速 Fast Path
    // 小矩阵可以完全装入 L1/L2 Cache，分块和 OpenMP 的线程唤醒开销反而会成为累赘
    if (N <= 128)
    {
        for (int i = 0; i < N; i += 4)
        {
            for (int j = 0; j < N; j += 8)
            {
                __m256 c0 = _mm256_loadu_ps(&C->data[(i + 0) * N + j]);
                __m256 c1 = _mm256_loadu_ps(&C->data[(i + 1) * N + j]);
                __m256 c2 = _mm256_loadu_ps(&C->data[(i + 2) * N + j]);
                __m256 c3 = _mm256_loadu_ps(&C->data[(i + 3) * N + j]);

                for (int k = 0; k < N; k++)
                {
                    __m256 b = _mm256_loadu_ps(&B->data[k * N + j]);
                    c0 = _mm256_add_ps(c0, _mm256_mul_ps(_mm256_set1_ps(A->data[(i + 0) * N + k]), b));
                    c1 = _mm256_add_ps(c1, _mm256_mul_ps(_mm256_set1_ps(A->data[(i + 1) * N + k]), b));
                    c2 = _mm256_add_ps(c2, _mm256_mul_ps(_mm256_set1_ps(A->data[(i + 2) * N + k]), b));
                    c3 = _mm256_add_ps(c3, _mm256_mul_ps(_mm256_set1_ps(A->data[(i + 3) * N + k]), b));
                }

                _mm256_storeu_ps(&C->data[(i + 0) * N + j], c0);
                _mm256_storeu_ps(&C->data[(i + 1) * N + j], c1);
                _mm256_storeu_ps(&C->data[(i + 2) * N + j], c2);
                _mm256_storeu_ps(&C->data[(i + 3) * N + j], c3);
            }
        }
        return 0;
    }

    // 设置缓存分块大小 (Tile Size/Block Size)，适合装入缓存
    const int BLOCK = 128;

    // 使用 OpenMP 对拆分后的 Block 进行多线程并行
#pragma omp parallel for collapse(2) default(none) shared(A, B, C, N,BLOCK)
    for (size_t i_b = 0; i_b < N; i_b += BLOCK)
    {
        for (size_t j_b = 0; j_b < N; j_b += BLOCK)
        {
            // 在 K 维度也进行分块，这是因为完全的缓存拦截需要三维分割
            for (size_t k_b = 0; k_b < N; k_b += BLOCK)
            {
                // 计算当前 Block 的边界
                size_t i_max = (i_b + BLOCK < N) ? i_b + BLOCK : N;
                size_t j_max = (j_b + BLOCK < N) ? j_b + BLOCK : N;
                size_t k_max = (k_b + BLOCK < N) ? k_b + BLOCK : N;

                // 寄存器分块 (Register Blocking)：一次处理一个 4x8 的小块
                for (size_t i = i_b; i < i_max; i += 4)
                {

                    for (size_t j = j_b; j < j_max; j += 8)
                    {
                        // 由于 k_b 切块了，C 的结果要分多次累加，因此需要 load 旧值而不是从 0 开始
                        __m256 c0 = _mm256_loadu_ps(&C->data[(i + 0) * N + j]);
                        __m256 c1 = _mm256_loadu_ps(&C->data[(i + 1) * N + j]);
                        __m256 c2 = _mm256_loadu_ps(&C->data[(i + 2) * N + j]);
                        __m256 c3 = _mm256_loadu_ps(&C->data[(i + 3) * N + j]);

                        for (size_t k = k_b; k < k_max; k++)
                        {
                            // 核心：复用加载进来的 B 矩阵的行片段，分别跟这 4 行的 A 元素相乘累加
                            __m256 b = _mm256_loadu_ps(&B->data[k * N + j]);
                            c0 = _mm256_add_ps(c0, _mm256_mul_ps(_mm256_set1_ps(A->data[(i + 0) * N + k]), b));
                            c1 = _mm256_add_ps(c1, _mm256_mul_ps(_mm256_set1_ps(A->data[(i + 1) * N + k]), b));
                            c2 = _mm256_add_ps(c2, _mm256_mul_ps(_mm256_set1_ps(A->data[(i + 2) * N + k]), b));
                            c3 = _mm256_add_ps(c3, _mm256_mul_ps(_mm256_set1_ps(A->data[(i + 3) * N + k]), b));
                        }

                        _mm256_storeu_ps(&C->data[(i + 0) * N + j], c0);
                        _mm256_storeu_ps(&C->data[(i + 1) * N + j], c1);
                        _mm256_storeu_ps(&C->data[(i + 2) * N + j], c2);
                        _mm256_storeu_ps(&C->data[(i + 3) * N + j], c3);
                    }
                }
            }
        }
    }

    return 0;
}


//=============================================OOC=================================================================


// 辅助函数：从硬盘的 N*N 矩阵中，读取一个左上角开头为 (row_start, col_start)，大小为 BLOCK*BLOCK 的块
void read_block_from_disk(FILE *f, Matrix const *block, size_t N, size_t row_start, size_t col_start)
{
    size_t B = block->rows; // BLOCK_SIZE
    for (size_t i = 0; i < B; i++)
    {
        // 计算在巨型矩阵中对应的绝对偏移量 (字节)
        size_t offset = ((row_start + i) * N + col_start) * sizeof(float);
        _fseeki64(f, offset, SEEK_SET); // Windows 下支持超大文件偏移的 fseek
        fread(&block->data[i * B], sizeof(float), B, f);
    }
}

// 辅助函数：将内存中的块写回硬盘的对应位置
void write_block_to_disk(FILE *f, Matrix const*block, size_t N, size_t row_start, size_t col_start)
{
    size_t B = block->rows; // BLOCK_SIZE
    for (size_t i = 0; i < B; i++)
    {
        size_t offset = ((row_start + i) * N + col_start) * sizeof(float);
        _fseeki64(f, offset, SEEK_SET);
        fwrite(&block->data[i * B], sizeof(float), B, f);
    }
}

// 核心：核外矩阵乘法
long long matmul_out_of_core(size_t N, int BLOCK_SIZE, const char *fileA, const char *fileB, const char *fileC)
{
    FILE *fA = fopen(fileA, "rb");
    FILE *fB = fopen(fileB, "rb");
    FILE *fC = fopen(fileC, "r+b"); // r+b 允许读写更新

    if (!fA || !fB || !fC)
    {
        printf("Failed to open matrix files.\n");
        return -1;
    }

    // 在内存中仅仅分配 3 个 BLOCK_SIZE * BLOCK_SIZE 的微型矩阵
    Matrix blockA = {BLOCK_SIZE, BLOCK_SIZE, (float *)malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(float))};
    Matrix blockB = {BLOCK_SIZE, BLOCK_SIZE, (float *)malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(float))};
    Matrix blockC = {BLOCK_SIZE, BLOCK_SIZE, (float *)malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(float))};

    long long start_time = get_time_ns();
    // 三层循环，遍历所有的宏块
    for (size_t i = 0; i < N; i += BLOCK_SIZE)
    {
        for (size_t j = 0; j < N; j += BLOCK_SIZE)
        {
            // 1. 先把原来 C 中该位置的底稿读入内存 (由于我们在 main 里面初始化了 C 为 0，第一次读进来的全是 0)
            read_block_from_disk(fC, &blockC, N, i, j);

            for (size_t k = 0; k < N; k += BLOCK_SIZE)
            {
                // 2. 读入 A 的宏块 和 B 的宏块
                read_block_from_disk(fA, &blockA, N, i, k);
                read_block_from_disk(fB, &blockB, N, k, j);

                // 3. 在内存中直接累加本次计算到 blockC
                // 由于 improved 本身的底层逻辑就是 c0 = _mm256_add_ps(c0, ...)
                // 它天然就会执行 C += A * B 的累加，因此不需要中转矩阵了！
                matmul_openblas_accumulate(BLOCK_SIZE, &blockA, &blockB, &blockC);
            }

            // 4. 这一大块 C 彻底算完了（所有 K 都加完了），写回硬盘！
            write_block_to_disk(fC, &blockC, N, i, j);
        }
        //printf("Out-of-Core Row %zu / %zu completed.\n", i, N);
    }
    long long end_time = get_time_ns();

    free(blockA.data);
    free(blockB.data);
    free(blockC.data);
    fclose(fA);
    fclose(fB);
    fclose(fC);
    return end_time - start_time;
}

long long type_switch(size_t n, enum TYPE_NAME type, const Matrix *A, const Matrix *B, Matrix const *C) {
    if (!A || !B || !C ||
        A->cols != n || A->rows != n ||
        B->cols != n || B->rows != n ||
        C->cols != n || C->rows != n ||
        !A->data || !B->data || !C->data ||
        n%16!=0 || n>20000 || n<=0
        )
    {
        printf("Invalid input for N = %zu\n", n);
        return -1; // Invalid input
    }
    clear_matrix(C);
    long long start = 0, end = 0;
    switch (type) {
        case IKJ:
            start = get_time_ns();
            matmul_ikj(n, A, B, C);
            end = get_time_ns();
            break;
        case IMPROVED:
            start = get_time_ns();
            matmul_improved(n, A, B, C);
            end = get_time_ns();
            break;
        case OPENBLAS:
            start = get_time_ns();
            matmul_openblas(n, A, B, C);
            end = get_time_ns();
            break;
        case TP:
            start = get_time_ns();
            matmul_tp(n, A, B, C);
            end = get_time_ns();
            break;
        case ALIGNED:
            start = get_time_ns();
            matmul_aligned(n, A, B, C);
            end = get_time_ns();
            break;
        case STRASSEN:
            start = get_time_ns();
            matmul_strassen(n, A, B, C);
            end = get_time_ns();
            break;
        case IMPROVED2:
            start = get_time_ns();
            matmul_improved2(n, A, B, C);
            end = get_time_ns();
            break;
        case IMPROVED3:
            start = get_time_ns();
            matmul_improved3(n, A, B, C);
            end = get_time_ns();
            break;
        default:
            start = get_time_ns();
            matmul_plain(n, A, B, C);
            end = get_time_ns();
            break;
    }
    return end - start;
}
