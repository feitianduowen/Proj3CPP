#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <omp.h>
#include "main.h"

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
int matmul_openblas(int N, const struct Matrix *A, const struct Matrix *B, struct Matrix *C)
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
