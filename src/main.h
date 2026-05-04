#ifndef MAIN_H
#define MAIN_H
#define _CRT_RAND_S
#include <stddef.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <math.h>

extern const char *fileA1;
extern const char *fileC1;
typedef struct
{
    size_t rows;
    size_t cols;
    float * data; // CPU memory
    // float * data_device; //GPU mememory
}Matrix;

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
enum TYPE_NAME {
    IKJ=0,
    IMPROVED=1,
    OPENBLAS=2,
    TP=3,
    ALIGNED=4,
    PLAIN=6,
    STRASSEN=7,
    IMPROVED2=8,
    IMPROVED3=9
};

long long get_time_ns();

void clear_matrix(Matrix const *mat);

int matmul_openblas(size_t N, const Matrix *A, const Matrix *B, Matrix const *C);
int matmul_openblas_accumulate(size_t N, const Matrix *A, const Matrix *B, Matrix const *C);

// 避免直接包含 OpenBLAS 头文件导致的 config.h 找不到问题，直接声明我们需要用到的 cblas_sgemm 函数

long long matmul_out_of_core(size_t N, int BLOCK_SIZE, const char *fileA, const char *fileB, const char *fileC);

void cblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const float alpha, const float *A,
                 const int lda, const float *B, const int ldb,
                 const float beta, float *C, const int ldc);

long long type_switch(size_t n, enum TYPE_NAME type, const Matrix *A, const Matrix *B, Matrix const *C);

#endif // MAIN_H
