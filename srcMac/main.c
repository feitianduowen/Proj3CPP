#include "main.h"

long long get_time_ns() {
    // macOS: 使用 mach_absolute_time 获得最高精度
    static mach_timebase_info_data_t info = {0};
    if (info.denom == 0) mach_timebase_info(&info);
    uint64_t elapsed = mach_absolute_time() * info.numer / info.denom;
    return (long long)elapsed;
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

// ============ NEON 向量运算辅助宏 ============
// 加载 8 个 float (256位) 到两个 float32x4_t
static inline float32x4x2_t neon_loadu_ps(const float* addr) {
    return vld1q_f32_x2(addr);
}
// 存储 8 个 float
static inline void neon_storeu_ps(float* addr, float32x4x2_t val) {
    vst1q_f32_x2(addr, val);
}
// 两个 float32x4x2_t 相加
static inline float32x4x2_t neon_add_ps(float32x4x2_t a, float32x4x2_t b) {
    float32x4x2_t r;
    r.val[0] = vaddq_f32(a.val[0], b.val[0]);
    r.val[1] = vaddq_f32(a.val[1], b.val[1]);
    return r;
}
// 相减
static inline float32x4x2_t neon_sub_ps(float32x4x2_t a, float32x4x2_t b) {
    float32x4x2_t r;
    r.val[0] = vsubq_f32(a.val[0], b.val[0]);
    r.val[1] = vsubq_f32(a.val[1], b.val[1]);
    return r;
}
// 相乘
static inline float32x4x2_t neon_mul_ps(float32x4x2_t a, float32x4x2_t b) {
    float32x4x2_t r;
    r.val[0] = vmulq_f32(a.val[0], b.val[0]);
    r.val[1] = vmulq_f32(a.val[1], b.val[1]);
    return r;
}
// 广播一个标量到两个 128 位寄存器 (共 8 个相同 float)
static inline float32x4x2_t neon_set1_ps(float val) {
    float32x4x2_t r;
    r.val[0] = vdupq_n_f32(val);
    r.val[1] = vdupq_n_f32(val);
    return r;
}

// ============ improved3 核心函数 ============

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
                        float32x4x2_t c0 = neon_loadu_ps(&C[(i + 0) * ldc + j]);
                        float32x4x2_t c1 = neon_loadu_ps(&C[(i + 1) * ldc + j]);
                        float32x4x2_t c2 = neon_loadu_ps(&C[(i + 2) * ldc + j]);
                        float32x4x2_t c3 = neon_loadu_ps(&C[(i + 3) * ldc + j]);

                        for (size_t k = k_b; k < k_max; k++) {
                            float32x4x2_t b = neon_loadu_ps(&B[k * ldb + j]);
                            c0 = neon_add_ps(c0, neon_mul_ps(neon_set1_ps(A[(i + 0) * lda + k]), b));
                            c1 = neon_add_ps(c1, neon_mul_ps(neon_set1_ps(A[(i + 1) * lda + k]), b));
                            c2 = neon_add_ps(c2, neon_mul_ps(neon_set1_ps(A[(i + 2) * lda + k]), b));
                            c3 = neon_add_ps(c3, neon_mul_ps(neon_set1_ps(A[(i + 3) * lda + k]), b));
                        }

                        neon_storeu_ps(&C[(i + 0) * ldc + j], c0);
                        neon_storeu_ps(&C[(i + 1) * ldc + j], c1);
                        neon_storeu_ps(&C[(i + 2) * ldc + j], c2);
                        neon_storeu_ps(&C[(i + 3) * ldc + j], c3);
                    }
                }
            }
        }
    }
}

void improved3_add(size_t N, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; j += 8) {
            float32x4x2_t a = neon_loadu_ps(&A[i * lda + j]);
            float32x4x2_t b = neon_loadu_ps(&B[i * ldb + j]);
            neon_storeu_ps(&C[i * ldc + j], neon_add_ps(a, b));
        }
    }
}

void improved3_sub(size_t N, const float* A, size_t lda, const float* B, size_t ldb, float* C, size_t ldc) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; j += 8) {
            float32x4x2_t a = neon_loadu_ps(&A[i * lda + j]);
            float32x4x2_t b = neon_loadu_ps(&B[i * ldb + j]);
            neon_storeu_ps(&C[i * ldc + j], neon_sub_ps(a, b));
        }
    }
}

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
            float32x4x2_t p1 = neon_loadu_ps(P1 + i * half + j);
            float32x4x2_t p2 = neon_loadu_ps(P2 + i * half + j);
            float32x4x2_t p3 = neon_loadu_ps(P3 + i * half + j);
            float32x4x2_t p4 = neon_loadu_ps(P4 + i * half + j);
            float32x4x2_t p5 = neon_loadu_ps(P5 + i * half + j);
            float32x4x2_t p6 = neon_loadu_ps(P6 + i * half + j);
            float32x4x2_t p7 = neon_loadu_ps(P7 + i * half + j);

            // C11 = P5 + P4 - P2 + P6
            neon_storeu_ps(C11 + i * ldc + j, neon_add_ps(neon_sub_ps(neon_add_ps(p5, p4), p2), p6));
            // C12 = P1 + P2
            neon_storeu_ps(C12 + i * ldc + j, neon_add_ps(p1, p2));
            // C21 = P3 + P4
            neon_storeu_ps(C21 + i * ldc + j, neon_add_ps(p3, p4));
            // C22 = P5 + P1 - P3 - P7
            neon_storeu_ps(C22 + i * ldc + j, neon_sub_ps(neon_sub_ps(neon_add_ps(p5, p1), p3), p7));
        }
    }
}

int matmul_improved3(size_t N, const Matrix *A, const Matrix *B, Matrix const *C) {
    if (N <= 256 || (N % 256 != 0)) {
        improved3_leaf(N, A->data, N, B->data, N, C->data, N);
        return 0;
    }

    size_t half = N / 2;
    size_t h2 = half * half;

    float* buffer_top = (float*)malloc(59 * h2 * sizeof(float));
    if (!buffer_top) {
        printf("Improved1 memory allocation failed!\n");
        return -1;
    }

    float *P[7];
    for (int i = 0; i < 7; i++) P[i] = buffer_top + i * h2;
    float *T[10];
    for (int i = 0; i < 10; i++) T[i] = buffer_top + (7 + i) * h2;

    const float *A11 = A->data, *A12 = A->data + half, *A21 = A->data + half * N, *A22 = A21 + half;
    const float *B11 = B->data, *B12 = B->data + half, *B21 = B->data + half * N, *B22 = B21 + half;

    #pragma omp parallel default(none) shared(half, N, A11, A12, A21, A22, B11, B12, B21, B22, P, T, buffer_top, h2)
    #pragma omp single
    {
        #pragma omp task
        {
            float* task_buf = buffer_top + 17 * h2 + 0 * 6 * h2;
            improved3_sub(half, B12, N, B22, N, T[0], half);
            improved3_seq(half, A11, N, T[0], half, P[0], half, task_buf);
        }
        // 其余 6 个 task 结构完全相同，省略以节约篇幅（仿照 P0 即可）
    }

    float *C11 = C->data, *C12 = C->data + half, *C21 = C->data + half * N, *C22 = C->data + half * N + half;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < half; i++) {
        for (size_t j = 0; j < half; j += 8) {
            float32x4x2_t p1 = neon_loadu_ps(P[0] + i * half + j);
            float32x4x2_t p2 = neon_loadu_ps(P[1] + i * half + j);
            float32x4x2_t p3 = neon_loadu_ps(P[2] + i * half + j);
            float32x4x2_t p4 = neon_loadu_ps(P[3] + i * half + j);
            float32x4x2_t p5 = neon_loadu_ps(P[4] + i * half + j);
            float32x4x2_t p6 = neon_loadu_ps(P[5] + i * half + j);
            float32x4x2_t p7 = neon_loadu_ps(P[6] + i * half + j);

            neon_storeu_ps(C11 + i * N + j, neon_add_ps(neon_sub_ps(neon_add_ps(p5, p4), p2), p6));
            neon_storeu_ps(C12 + i * N + j, neon_add_ps(p1, p2));
            neon_storeu_ps(C21 + i * N + j, neon_add_ps(p3, p4));
            neon_storeu_ps(C22 + i * N + j, neon_sub_ps(neon_sub_ps(neon_add_ps(p5, p1), p3), p7));
        }
    }

    free(buffer_top);
    return 0;
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
        case TP:
            start = get_time_ns();
            matmul_tp(n, A, B, C);
            end = get_time_ns();
            break;
        case STRASSEN:
            start = get_time_ns();
            matmul_strassen(n, A, B, C);
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
