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

struct Matrix *create_matrix_aligned(size_t rows, size_t cols)
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
    mat->data = (float *)_aligned_malloc(rows * cols * sizeof(float), 32);
    if (!mat->data)
    {
        free(mat);
        return NULL;
    }
    return mat;
}

void free_matrix_aligned(struct Matrix *mat)
{
    if (mat)
    {
        if (mat->data) _aligned_free(mat->data);
        free(mat);
    }
}

void randomize_matrix(struct Matrix const *mat)
{
    if (!mat || !mat->data) return;
    size_t count = mat->rows * mat->cols;
    for (size_t i = 0; i < count; i++)
        mat->data[i] = (float)rand() / RAND_MAX;
}

void clear_matrix(struct Matrix const *mat)
{
    if (!mat || !mat->data) return;
    size_t count = mat->rows * mat->cols;
    for (size_t i = 0; i < count; i++)
        mat->data[i] = 0.0f;
}

int compare_matrices(const struct Matrix *mat1, const struct Matrix *mat2, float epsilon)
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
        //printf("  Max error: %e\n", max_err);
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

int matmul_plain(size_t N, const struct Matrix *A, const struct Matrix *B, struct Matrix const *C)
{
    if (!A || !B || !C || N <= 0 ||
        A->cols != N || A->rows != N ||
        B->cols != N || B->rows != N ||
        C->cols != N || C->rows != N ||
        !A->data || !B->data || !C->data)
    {
        return -1; // Invalid input
    }
    clear_matrix(C);
    
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

int matmul_ikj(size_t N, const struct Matrix *A, const struct Matrix *B, struct Matrix const *C)
{
    if (!A || !B || !C || N <= 0 ||
        A->cols != N || A->rows != N ||
        B->cols != N || B->rows != N ||
        C->cols != N || C->rows != N ||
        !A->data || !B->data || !C->data)
    {
        return -1; // Invalid input
    }
    
    clear_matrix(C);
    
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

int matmul_improved(size_t N, const struct Matrix *A, const struct Matrix *B, struct Matrix const *C)
{
    if (N>10000)
    {
        printf("Matrix size N = %zu is too large for in-memory multiplication.\n", N);
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
    
    // Fallback for non-multiples of 8 to avoid SIMD out-of-bounds heap corruption
    if (N % 8 != 0) {
        return matmul_ikj(N, A, B, C);
    }

    clear_matrix(C);

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

int matmul_improved_aligned(size_t N, const struct Matrix *A, const struct Matrix *B, struct Matrix const *C)
{
    if (N>10000)
    {
        printf("Matrix size N = %zu is too large for in-memory multiplication.\n", N);
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
    
    // Fallback for non-multiples of 8 to avoid SIMD out-of-bounds
    if (N % 8 != 0) {
        return matmul_ikj(N, A, B, C);
    }

    clear_matrix(C);

    if (N <= 128)
    {
        for (int i = 0; i < N; i += 4)
        {
            for (int j = 0; j < N; j += 8)
            {
                __m256 c0 = _mm256_load_ps(&C->data[(i + 0) * N + j]);
                __m256 c1 = _mm256_load_ps(&C->data[(i + 1) * N + j]);
                __m256 c2 = _mm256_load_ps(&C->data[(i + 2) * N + j]);
                __m256 c3 = _mm256_load_ps(&C->data[(i + 3) * N + j]);

                for (int k = 0; k < N; k++)
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

#pragma omp parallel for collapse(2)
    for (int i_b = 0; i_b < N; i_b += BLOCK)
    {
        for (int j_b = 0; j_b < N; j_b += BLOCK)
        {
            for (int k_b = 0; k_b < N; k_b += BLOCK)
            {
                int i_max = (i_b + BLOCK < N) ? i_b + BLOCK : N;
                int j_max = (j_b + BLOCK < N) ? j_b + BLOCK : N;
                int k_max = (k_b + BLOCK < N) ? k_b + BLOCK : N;

                for (int i = i_b; i < i_max; i += 4)
                {
                    for (int j = j_b; j < j_max; j += 8)
                    {
                        __m256 c0 = _mm256_load_ps(&C->data[(i + 0) * N + j]);
                        __m256 c1 = _mm256_load_ps(&C->data[(i + 1) * N + j]);
                        __m256 c2 = _mm256_load_ps(&C->data[(i + 2) * N + j]);
                        __m256 c3 = _mm256_load_ps(&C->data[(i + 3) * N + j]);

                        for (int k = k_b; k < k_max; k++)
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

int matmul_openblas(size_t N, const struct Matrix *A, const struct Matrix *B, struct Matrix const *C)
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

int matmul_openblas_accumulate(size_t N, const struct Matrix *A, const struct Matrix *B, struct Matrix const *C)
{
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N,
                1.0f, A->data, N,
                B->data, N,
                1.0f, C->data, N); // <--- 这里改成 1.0f 就自带累加了！
    return 0;
}

int matmul_tp(size_t N, const struct Matrix *A, const struct Matrix *B, struct Matrix const *C)
{
    if (N > 10000)
    {
        printf("Matrix size N = %zu is too large for in-memory multiplication.\n", N);
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
    
    // Fallback for non-multiples of 8 to avoid SIMD out-of-bounds heap corruption
    if (N % 8 != 0) {
        return matmul_ikj(N, A, B, C);
    }
    
    clear_matrix(C);

    // 针对小矩阵 (N <= 128) 的特化全速 Fast Path
    // 引入 i-k-j 优化
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
                    // 将 A 提取到外层，避免在 j 循环中重复 load
                    __m256 a0 = _mm256_set1_ps(A->data[(i + 0) * N + k]);
                    __m256 a1 = _mm256_set1_ps(A->data[(i + 1) * N + k]);
                    __m256 a2 = _mm256_set1_ps(A->data[(i + 2) * N + k]);
                    __m256 a3 = _mm256_set1_ps(A->data[(i + 3) * N + k]);

                    // 此时 B 的访问完全连续
                    __m256 b = _mm256_loadu_ps(&B->data[k * N + j]);

                    // 分别与 A 的四行元素相乘，并累加到 C
                    c0 = _mm256_add_ps(c0, _mm256_mul_ps(a0, b));
                    c1 = _mm256_add_ps(c1, _mm256_mul_ps(a1, b));
                    c2 = _mm256_add_ps(c2, _mm256_mul_ps(a2, b));
                    c3 = _mm256_add_ps(c3, _mm256_mul_ps(a3, b));
                }
                
                _mm256_storeu_ps(&C->data[(i + 0) * N + j], c0);
                _mm256_storeu_ps(&C->data[(i + 1) * N + j], c1);
                _mm256_storeu_ps(&C->data[(i + 2) * N + j], c2);
                _mm256_storeu_ps(&C->data[(i + 3) * N + j], c3);
            }
        }
        return 0;
    }

    // 设置缓存分块大小 (Tile Size/Block Size)
    const int BLOCK = 128;

    // 外层并行分配：因为是对 C 矩阵的输出区块划分，所以必须保留 i_b 和 j_b 作为外层
#pragma omp parallel for collapse(2)
    for (int i_b = 0; i_b < N; i_b += BLOCK)
    {
        for (int j_b = 0; j_b < N; j_b += BLOCK)
        {
            for (int k_b = 0; k_b < N; k_b += BLOCK)
            {
                int i_max = (i_b + BLOCK < N) ? i_b + BLOCK : N;
                int j_max = (j_b + BLOCK < N) ? j_b + BLOCK : N;
                int k_max = (k_b + BLOCK < N) ? k_b + BLOCK : N;

                // 寄存器分块
                for (int i = i_b; i < i_max; i += 4)
                {
                    for (int j = j_b; j < j_max; j += 8)
                    {
                        __m256 c0 = _mm256_loadu_ps(&C->data[(i + 0) * N + j]);
                        __m256 c1 = _mm256_loadu_ps(&C->data[(i + 1) * N + j]);
                        __m256 c2 = _mm256_loadu_ps(&C->data[(i + 2) * N + j]);
                        __m256 c3 = _mm256_loadu_ps(&C->data[(i + 3) * N + j]);

                        // 核心修改：内层循环顺序保持 k -> j 已经被外层 j 给包裹了，所以只能放里面让C累加
                        for (int k = k_b; k < k_max; k++)
                        {
                            // 预先准备好当前 4 行在第 k 列的 A 元素标量向量
                            __m256 a0 = _mm256_set1_ps(A->data[(i + 0) * N + k]);
                            __m256 a1 = _mm256_set1_ps(A->data[(i + 1) * N + k]);
                            __m256 a2 = _mm256_set1_ps(A->data[(i + 2) * N + k]);
                            __m256 a3 = _mm256_set1_ps(A->data[(i + 3) * N + k]);

                            // 此处的 B 连续加载将极大提高 L1/L2 Cache 吞吐量
                            __m256 b = _mm256_loadu_ps(&B->data[k * N + j]);

                            c0 = _mm256_add_ps(c0, _mm256_mul_ps(a0, b));
                            c1 = _mm256_add_ps(c1, _mm256_mul_ps(a1, b));
                            c2 = _mm256_add_ps(c2, _mm256_mul_ps(a2, b));
                            c3 = _mm256_add_ps(c3, _mm256_mul_ps(a3, b));
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

long long test_outer(size_t n, size_t innerCir, size_t outerCir,int type)
{
    long long res=LONG_LONG_MAX,tp=0;
    struct Matrix *A,*C;
    if (type==4) {
        A = create_matrix_aligned(n, n);
        C = create_matrix_aligned(n, n);
    }else {
        A = create_matrix(n, n);
        C = create_matrix(n, n);
    }


    for (size_t i = 0; i < outerCir; i++)
    {
        clear_matrix(A);
        randomize_matrix(A);
        clear_matrix(C);
        tp = test_inner(n, innerCir, type, A, C);

        if (tp == -1) {
            printf("Test failed for N = %zu in outer loop %zu\n", n, i);
            free_matrix(A);
            free_matrix(C);
            return -1;
        }
        if (tp<res) res=tp;

    }
    if (type==4) {
        free_matrix_aligned(A);
        free_matrix_aligned(C);
    } else {
        free_matrix(A);
        free_matrix(C);
    }
    return res;
}

long long test_inner(size_t n, size_t innerCir, int type, const struct Matrix *A, struct Matrix const *C) {
    long long time_res = LONG_LONG_MAX, start = 0, end = 0;
    if (!A ||!C)
    {
        printf("Memory allocation failed for N = %zu\n", n);
        return -1;
    }

        for (size_t j = 0; j < innerCir; j++)
        {
            switch (type) {
                case 0:
                    start = get_time_ns();
                    matmul_ikj(n, A, A, C);
                    end = get_time_ns();
                    break;
                case 1:
                    start = get_time_ns();
                    matmul_improved(n, A, A, C);
                    end = get_time_ns();
                    break;
                case 2:
                    start = get_time_ns();
                    matmul_openblas(n, A, A, C);
                    end = get_time_ns();
                    break;
                case 3:
                    start = get_time_ns();
                    matmul_tp(n, A, A, C);
                    end = get_time_ns();
                    break;
                case 4:
                    start = get_time_ns();
                    matmul_improved_aligned(n, A, A, C);
                    end = get_time_ns();
                    break;
                default:
                    start = get_time_ns();
                    matmul_plain(n, A, A, C);
                    end = get_time_ns();
                    break;
            }
            if (end - start < time_res)
                time_res = end - start;
        }
    return time_res;
}

int check_result(size_t sizeN, size_t cir,int type,float epsilon)
{
    printf("Checking correctness for size N = %zu with %zu iterations...\n", sizeN, cir);
    struct Matrix *A = create_matrix(sizeN, sizeN);
    struct Matrix *B = create_matrix(sizeN, sizeN);
    struct Matrix *C1 = create_matrix(sizeN, sizeN);
    struct Matrix *C2 = create_matrix(sizeN, sizeN);
    int result = 0;
    if (!A || !B || !C1 || !C2)
    {
        printf("Memory allocation failed for check test\n");
        free_matrix(A);
        free_matrix(B);
        free_matrix(C1);
        free_matrix(C2);
        return -1;
    }
    for (size_t i = 0; i < cir; i++)
    {

        randomize_matrix(A);
        randomize_matrix(B);

        matmul_plain(sizeN, A, B, C1);
        switch (type) {
            case 0:
                matmul_ikj(sizeN, A, B, C2);
                break;
            case 1:
                matmul_improved(sizeN, A, B, C2);
                break;
            default:
                matmul_improved(sizeN, A, B, C2);

        }

        result = compare_matrices(C1, C2, epsilon);
        if (!result)
        {
            printf("Result mismatch for N = %zu\n", sizeN);
            break;
        }
    }

    free_matrix(A);
    free_matrix(B);
    free_matrix(C1);
    free_matrix(C2);
    printf("Check for N = %zu %s\n", sizeN, result ? "PASSED" : "FAILED");
    return result;
}