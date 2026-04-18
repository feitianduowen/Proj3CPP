#ifndef MAIN_H
#define MAIN_H

#include <stddef.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>

struct Matrix
{
    size_t rows;
    size_t cols;
    float *data;
};

struct TestResult {
    int size;
    double time_plain;
    double time_improved;
    double time_openblas;
};

long long get_time_ns();

int matmul_plain(int N, const struct Matrix *A, const struct Matrix *B, struct Matrix *C);
int matmul_improved(int N, const struct Matrix *A, const struct Matrix *B, struct Matrix *C);
int matmul_openblas(size_t N, const struct Matrix *A, const struct Matrix *B, struct Matrix *C);
int matmul_openblas_accumulate(int N, const struct Matrix *A, const struct Matrix *B, struct Matrix const*C);

struct Matrix *create_matrix(size_t rows, size_t cols);
void randomize_matrix(struct Matrix *mat);
void clear_matrix(struct Matrix *mat);
void free_matrix(struct Matrix *mat);
int compare_matrices(struct Matrix *mat1, struct Matrix *mat2,float epsilon);
long long run_test(int N, int innerCir, const struct Matrix *A, const struct Matrix *B, struct Matrix *C, int type);

// 避免直接包含 OpenBLAS 头文件导致的 config.h 找不到问题，直接声明我们需要用到的 cblas_sgemm 函数
enum CBLAS_ORDER
{
    CblasRowMajor = 101,
    CblasColMajor = 102
};
enum CBLAS_TRANSPOSE
{
    CblasNoTrans = 111,
    CblasTrans = 112,
    CblasConjTrans = 113
};

void cblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const float alpha, const float *A,
                 const int lda, const float *B, const int ldb,
                 const float beta, float *C, const int ldc);

#endif // MAIN_H
