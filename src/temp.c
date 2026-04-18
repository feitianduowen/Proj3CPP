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

int record_plain()
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

int check_ikj(size_t N,int cir)
{
    long long time_plain = LONG_LONG_MAX, time_ikj = LONG_LONG_MAX, start = 0, end = 0;
     struct Matrix *A = create_matrix(N, N);
    struct Matrix *B = create_matrix(N, N);
    struct Matrix *C1 = create_matrix(N, N);
    struct Matrix *C2 = create_matrix(N, N);
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

        start = get_time_ns();
        matmul_plain(N, A, B, C1);
        end = get_time_ns();
        if(end-start<time_plain)time_plain = end - start;

        start = get_time_ns();
        matmul_ikj(N, A, B, C2);
        end = get_time_ns();
        if(end-start<time_ikj)time_ikj = end - start;

        result = compare_matrices(C1, C2, 1e-3f);
        if (!result)
        {
            printf("Result mismatch for N = %zu\n", N);
            break;
        }
    }

    free_matrix(A);
    free_matrix(B);
    free_matrix(C1);
    free_matrix(C2);
    printf("plain: %lld ns, ikj: %lld ns\n", time_plain, time_ikj );
    printf("Check for N = %zu %s\n", N, result ? "PASSED" : "FAILED");
    return result;
}
int main(){
    // check_ikj(16,10000);
    // Sleep(1000);
    // check_ikj(50,10000);
    // Sleep(10000);
    // check_ikj(128,1000);
    // Sleep(1000);
    // check_ikj(256,500);
    // Sleep(1000);
    // check_ikj(400,1000);
    // Sleep(1000);
    // check_ikj(800,100);
    // Sleep(1000);
     check_ikj(1600,10);
    // Sleep(1000);
    //check_ikj(2000,5);
    return 0;
}