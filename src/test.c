#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "main.h"

int sizes[] = {16, 128, 400, 800, 1024, 8192, 64000};
int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

long long get_time_ns()
{
    struct timespec ts;
    // C11 标准时间函数，能获取纳秒精度（在 GCC/MinGW 下完美支持）
    timespec_get(&ts, TIME_UTC);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

struct Matrix *create_matrix(size_t rows, size_t cols)
{
    struct Matrix *mat = (struct Matrix *)malloc(sizeof(struct Matrix));
    if (!mat)
        return NULL;
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
        if (mat->data)
            free(mat->data);
        free(mat);
    }
}

void randomize_matrix(struct Matrix *mat)
{
    if (!mat || !mat->data)
        return;
    size_t count = mat->rows * mat->cols;
    for (size_t i = 0; i < count; i++)
    {
        mat->data[i] = (float)rand() / RAND_MAX;
    }
}

void clear_matrix(struct Matrix *mat)
{
    if (!mat || !mat->data)
        return;
    size_t count = mat->rows * mat->cols;
    for (size_t i = 0; i < count; i++)
    {
        mat->data[i] = 0.0f;
    }
}

int compare_matrices(struct Matrix *mat1, struct Matrix *mat2)
{
    if (!mat1 || !mat2 || mat1->rows != mat2->rows || mat1->cols != mat2->cols)
        return 0;
    size_t count = mat1->rows * mat1->cols;
    for (size_t i = 0; i < count; i++)
    {
        if (fabs(mat1->data[i] - mat2->data[i]) > 1e-3)
        {
            return 0; // differ
        }
    }
    return 1; // match
}

double run_test(int N, int innerCir, const struct Matrix *A, const struct Matrix *B, struct Matrix *C, const int type)
{
    if (!A || !B || !C)
    {
        printf("Memory allocation failed for N = %d\n", N);
        return -1;
    }

    clear_matrix(C);

    long long time_cum = 0, start = 0, end = 0;
    double time_avg = 0.0;
    switch (type)
    {
    case 0:
        for (size_t i = 0; i < innerCir; i++)
        {
            start = get_time_ns();
            matmul_plain(N, A, B, C);
            end = get_time_ns();
            time_cum += (end - start);
        }

        break;
    case 1:
        for (size_t i = 0; i < innerCir; i++)
        {
            start = get_time_ns();
            matmul_improved(N, A, B, C);
            end = get_time_ns();
            time_cum += (end - start);
        }

        break;
    default:
        for (size_t i = 0; i < innerCir; i++)
        {
            start = get_time_ns();
            matmul_openblas(N, A, B, C);
            end = get_time_ns();
            time_cum += (end - start);
        }
        break;
    }

    time_avg = (double)time_cum / innerCir;

    return time_avg;
}

void test(int sizeN, int cir0, int cir1, int cir2, int outerCir, struct TestResult *res)
{
    printf("Testing size N = %d with %d outer iterations...\n", sizeN, outerCir);
    struct Matrix *A = create_matrix(sizeN, sizeN);
    struct Matrix *B = create_matrix(sizeN, sizeN);
    struct Matrix *C = create_matrix(sizeN, sizeN);
    if (!A || !B || !C)
    {
        printf("Memory allocation failed for N = %d\n", sizeN);
        return;
    }
    // 注意：每次测试前重新置零，否则之前的计算结果会留在 C 里
    // 下面调用的 run_test() 内部在开头已经固定调用了 clear_matrix(C);
    // 因此这里外层的 clear_matrix(C) 是多余的，但保留也不会错。
    
    double time_plain = 0.0, time_improved = 0.0, time_openblas = 0.0;
    for (size_t i = 0; i < outerCir; i++)
    {
        randomize_matrix(A);
        randomize_matrix(B);
        
        time_plain += run_test(sizeN, cir0, A, B, C, 0);
        time_improved += run_test(sizeN, cir1, A, B, C, 1);
        time_openblas += run_test(sizeN, cir2, A, B, C, 2);
    }

    if (cir0 != 0)
        time_plain /= outerCir;
    time_improved /= outerCir;
    time_openblas /= outerCir;

    if (res)
    {
        res->size = sizeN;
        if (cir0 != 0)res->time_plain = time_plain;
        res->time_improved = time_improved;
        res->time_openblas = time_openblas;
    }

    printf("Size: %d | Plain: %.2f ns | Improved: %.2f ns | OpenBLAS: %.2f ns\n",
           sizeN,
           time_plain,
           time_improved,
           time_openblas);

    free_matrix(A);
    free_matrix(B);
    free_matrix(C);
}

int check(int sizeN, int cir)
{
    printf("Checking correctness for size N = %d with %d iterations...\n", sizeN, cir);
    struct Matrix *A = create_matrix(sizeN, sizeN);
    struct Matrix *B = create_matrix(sizeN, sizeN);
    struct Matrix *C1 = create_matrix(sizeN, sizeN);
    struct Matrix *C2 = create_matrix(sizeN, sizeN);
    int result = 0;
    if (!A || !B || !C1 || !C2)
    {
        printf("Memory allocation failed for check test\n");
        return -1;
    }
    for (size_t i = 0; i < cir; i++)
    {

        randomize_matrix(A);
        randomize_matrix(B);

        matmul_plain(sizeN, A, B, C1);
        matmul_openblas(sizeN, A, B, C2);

        result = compare_matrices(C1, C2);
        if (!result)
        {
            printf("Result mismatch for N = %d\n", sizeN);
            break;
        }
    }

    free_matrix(A);
    free_matrix(B);
    free_matrix(C1);
    free_matrix(C2);
    printf("Check for N = %d %s\n", sizeN, result ? "PASSED" : "FAILED");
    return result;
}

int main()
{
    srand((unsigned int)time(NULL));
    // 64K is too large for normal memory allocation
    // check(16, 10000);
    // check(128, 1000);
    // check(400, 100);
    // check(800, 10);
    // check(1024, 3);
    // check(8192, 1);

    struct TestResult results[7];
    int res_idx = 0;

    test(16, 0, 10, 10, 1000, &results[res_idx++]);
    test(128, 0, 10, 10, 1000, &results[res_idx++]);
    test(400, 0, 10, 10, 1000, &results[res_idx++]);
    test(800, 0, 10, 10, 1000, &results[res_idx++]);
    test(1024, 0, 1, 1, 5, &results[res_idx++]);
    test(8192, 0, 1, 1, 5, &results[res_idx++]);
    test(64000, 0, 1, 1, 3, &results[res_idx++]);

    FILE *fp = fopen("result.csv", "w");
    if (fp)
    {
        fprintf(fp, "Size,Plain(ns),Improved(ns),OpenBLAS(ns)\n");
        for (int i = 0; i < res_idx; i++)
        {
            fprintf(fp, "%d,%.2f,%.2f,%.2f\n", results[i].size, results[i].time_plain, results[i].time_improved, results[i].time_openblas);
        }
        fclose(fp);
        printf("\n==> Results successfully saved to result.csv\n");
    }
    else
    {
        printf("\n==> Failed to open result.csv for writing.\n");
    }

    return 0;
}
