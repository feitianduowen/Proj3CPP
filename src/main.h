#ifndef MAIN_H
#define MAIN_H

#include <stddef.h>

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


int matmul_plain(int N, const struct Matrix *A, const struct Matrix *B, struct Matrix *C);
int matmul_improved(int N, const struct Matrix *A, const struct Matrix *B, struct Matrix *C);
int matmul_openblas(int N, const struct Matrix *A, const struct Matrix *B, struct Matrix *C);

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
