#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "main.h"

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

void free_matrix(struct Matrix *mat)
{
    if (mat)
    {
        if (mat->data)
            free(mat->data);
        free(mat);
    }
}

long long tt(int n)
{
    struct Matrix *A = create_matrix(n, n);
    struct Matrix *B = create_matrix(n, n);
    struct Matrix *C = create_matrix(n, n);
    if (!A || !B || !C)
    {
        printf("Memory allocation failed for N = %d\n", n);
        return -1;
    }
    randomize_matrix(A);
    randomize_matrix(B);

    long long start = get_time_ns();
    matmul_plain(n, A, B, C);
    long long end = get_time_ns();

    free_matrix(A);
    free_matrix(B);
    free_matrix(C);

    return end - start;
}

int main()
{
    int sizes[] = {16, 128, 400, 800, 1024, 2000, 8192, 64000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    FILE *fp = fopen("plot.csv", "w");
    if (!fp) {
        printf("Failed to open plot.csv for writing\n");
        return 1;
    }
    fprintf(fp, "Size,Time(ms)\n");

    for (int i = 16; i < 1600; i+=16)
    {
        long long time_plain_cum = 0;
        for (int j = 0; j < 100; j++) // Run 100 times for each size
        {
            long long time_plain = tt(i);
            time_plain_cum += time_plain;
        }
        //printf("Size: %d, Time: %.3f ms\n", i, time_plain_cum / 100.0 / 1e6);
        fprintf(fp, "%d,%.3f\n", i, time_plain_cum / 100.0 / 1e6);
    }

    fclose(fp);
    return 0;
}
