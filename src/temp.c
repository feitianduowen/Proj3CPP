#include "main.h"

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
