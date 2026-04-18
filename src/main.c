#include "main.h"

struct Matrix *create_matrix(size_t rows, size_t cols)
{
    if (rows>10000||cols>10000)
    {
        printf("Matrix size N = %zu is too large for in-memory handling.\n", rows);
        return NULL;
    }
    struct Matrix *mat = (struct Matrix *)malloc(sizeof(struct Matrix));
    if (!mat) return NULL;
    mat->rows = rows;
    mat->cols = cols;
    mat->data = (float *)malloc(rows * cols * sizeof(float));
    if (!mat->data)
    {
        free(mat);
        return NULL;
    }
    return mat;
}

void free_matrix(struct Matrix *mat)
{
    if (mat)
    {
        if (mat->data) free(mat->data);
        free(mat);
    }
}

void randomize_matrix(struct Matrix *mat)
{
    if (!mat || !mat->data) return;
    size_t count = mat->rows * mat->cols;
    for (size_t i = 0; i < count; i++)
        mat->data[i] = (float)rand() / RAND_MAX;
}

void clear_matrix(struct Matrix *mat)
{
    if (!mat || !mat->data) return;
    size_t count = mat->rows * mat->cols;
    for (size_t i = 0; i < count; i++)
        mat->data[i] = 0.0f;
}

int compare_matrices(struct Matrix *mat1, struct Matrix *mat2, float epsilon)
{
    if (!mat1 || !mat2 || mat1->rows != mat2->rows || mat1->cols != mat2->cols) return 0;
    size_t count = mat1->rows * mat1->cols;
    int passed = 1;
    float max_err = 0.0f,diff=0.0f;
    
    for (size_t i = 0; i < count; i++)
    {
        diff = (float)fabs(mat1->data[i] - mat2->data[i]);
        if (diff > max_err)
            max_err = diff;
        if (diff > epsilon) 
        {
            if (passed) {
                printf("  Mismatch at index %zu: mat1=%.4f vs mat2=%.4f\n", i, mat1->data[i], mat2->data[i]);
            }
            passed = 0;
        }
    }
    if (passed) {
        printf("  Max error: %e\n", max_err);
    }
    return passed;
}

long long get_time_ns()
{
    struct timespec ts;
    // C11 标准时间函数，能获取纳秒精度（在 GCC/MinGW 下完美支持）
    timespec_get(&ts, TIME_UTC);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

int matmul_plain(int N, const struct Matrix *A, const struct Matrix *B, struct Matrix *C)
{
    if (!A || !B || !C || N <= 0 ||
        A->cols != N || A->rows != N ||
        B->cols != N || B->rows != N ||
        C->cols != N || C->rows != N ||
        !A->data || !B->data || !C->data)
    {
        return -1; // Invalid input
    }

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
int matmul_improved(int N, const struct Matrix *A, const struct Matrix *B, struct Matrix *C)
{
    if (N>10000)
    {
        printf("Matrix size N = %d is too large for in-memory multiplication.\n", N);
        return -1;
    }
    
    if (!A || !B || !C || N <= 0 ||
        A->cols != N || A->rows != N ||
        B->cols != N || B->rows != N ||
        C->cols != N || C->rows != N ||
        !A->data || !B->data || !C->data)
    {
        return -1; // Invalid input
    }

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
#pragma omp parallel for collapse(2)
    for (int i_b = 0; i_b < N; i_b += BLOCK)
    {
        for (int j_b = 0; j_b < N; j_b += BLOCK)
        {
            // 在 K 维度也进行分块，这是因为完全的缓存拦截需要三维分割
            for (int k_b = 0; k_b < N; k_b += BLOCK)
            {
                // 计算当前 Block 的边界
                int i_max = (i_b + BLOCK < N) ? i_b + BLOCK : N;
                int j_max = (j_b + BLOCK < N) ? j_b + BLOCK : N;
                int k_max = (k_b + BLOCK < N) ? k_b + BLOCK : N;

                // 寄存器分块 (Register Blocking)：一次处理一个 4x8 的小块
                for (int i = i_b; i < i_max; i += 4)
                {

                    for (int j = j_b; j < j_max; j += 8)
                    {
                        // 由于 k_b 切块了，C 的结果要分多次累加，因此需要 load 旧值而不是从 0 开始
                        __m256 c0 = _mm256_loadu_ps(&C->data[(i + 0) * N + j]);
                        __m256 c1 = _mm256_loadu_ps(&C->data[(i + 1) * N + j]);
                        __m256 c2 = _mm256_loadu_ps(&C->data[(i + 2) * N + j]);
                        __m256 c3 = _mm256_loadu_ps(&C->data[(i + 3) * N + j]);

                        for (int k = k_b; k < k_max; k++)
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
int matmul_openblas(size_t N, const struct Matrix *A, const struct Matrix *B, struct Matrix *C)
{
    if (!A || !B || !C || N <= 0)
        return -1;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N,
                1.0f, A->data, N,
                B->data, N,
                0.0f, C->data, N);
    return 0;
}

int matmul_openblas_accumulate(int N, const struct Matrix *A, const struct Matrix *B, struct Matrix const *C)
{
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N,
                1.0f, A->data, N,
                B->data, N,
                1.0f, C->data, N); // <--- 这里改成 1.0f 就自带累加了！
    return 0;
}