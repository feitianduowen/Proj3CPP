#ifndef MAIN_H
#define MAIN_H

#include <stddef.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <math.h>
struct Matrix
{
    size_t rows;
    size_t cols;
    float *data;
};

struct TestResult {
    int size;
    long long time_plain;
    long long time_improved;
    long long time_openblas;
};

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

long long get_time_ns();

int matmul_plain(size_t N, const struct Matrix *A, const struct Matrix *B, struct Matrix const *C);
int matmul_ikj(size_t N, const struct Matrix *A, const struct Matrix *B, struct Matrix const *C);
int matmul_improved(size_t N, const struct Matrix *A, const struct Matrix *B, struct Matrix const *C);
int matmul_openblas(size_t N, const struct Matrix *A, const struct Matrix *B, struct Matrix const *C);
int matmul_openblas_accumulate(size_t N, const struct Matrix *A, const struct Matrix *B, struct Matrix const*C);
int matmul_tp(size_t N, const struct Matrix *A, const struct Matrix *B, struct Matrix const *C);


struct Matrix *create_matrix(size_t rows, size_t cols);
void randomize_matrix(struct Matrix const *mat);
void clear_matrix(struct Matrix const *mat);
void free_matrix(struct Matrix *mat);
int compare_matrices(const struct Matrix *mat1, const struct Matrix *mat2, float epsilon);

long long test_outer(size_t n, size_t innerCir, size_t outerCir,int type);
long long test_inner(size_t n, size_t innerCir, int type, const struct Matrix *A, struct Matrix const *C);
long long run_test(int N, int innerCir, const struct Matrix *A, const struct Matrix *B, struct Matrix *C, int type);
int check_result(size_t sizeN, size_t cir,int type,float epsilon);
// 避免直接包含 OpenBLAS 头文件导致的 config.h 找不到问题，直接声明我们需要用到的 cblas_sgemm 函数

void cblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const float alpha, const float *A,
                 const int lda, const float *B, const int ldb,
                 const float beta, float *C, const int ldc);

#endif // MAIN_H
