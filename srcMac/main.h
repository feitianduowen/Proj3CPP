#ifndef MAIN_H
#define MAIN_H
#include <mach/mach_time.h>
#include <string.h>
#include <arm_neon.h>
#include <stddef.h>
#include <limits.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>

extern const char *fileA1;
extern const char *fileC1;

typedef struct {
    size_t rows;
    size_t cols;
    float *data;
} Matrix;

struct TestResult {
    int size;
    long long time_plain;
    long long time_improved;
    long long time_openblas;
};

enum CBLAS_ORDER {
    CblasRowMajor = 101,
    CblasColMajor = 102
};
enum CBLAS_TRANSPOSE {
    CblasNoTrans = 111,
    CblasTrans = 112,
    CblasConjTrans = 113
};
enum TYPE_NAME {
    IKJ = 0,
    IMPROVED = 1,
    OPENBLAS = 2,
    TP = 3,
    ALIGNED = 4,
    PLAIN = 6,
    STRASSEN = 7,
    IMPROVED2 = 8,
    IMPROVED3 = 9
};

long long get_time_ns();
void clear_matrix(const Matrix *mat);
int matmul_openblas(size_t N, const Matrix *A, const Matrix *B, const Matrix *C);
int matmul_openblas_accumulate(size_t N, const Matrix *A, const Matrix *B, const Matrix *C);
long long matmul_out_of_core(size_t N, int BLOCK_SIZE, const char *fileA, const char *fileB, const char *fileC);


long long type_switch(size_t n, enum TYPE_NAME type, const Matrix *A, const Matrix *B, const Matrix *C);

#endif