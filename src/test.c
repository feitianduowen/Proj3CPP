#include "main.h"

struct TestResult results[7];

long long run_test(int N, int innerCir, const struct Matrix *A, const struct Matrix *B, struct Matrix *C, int type)
{
    if (!A || !B || !C)
    {
        printf("Memory allocation failed for N = %d\n", N);
        return -1;
    }

    clear_matrix(C);

    long long time_min = LONG_LONG_MAX, start = 0, end = 0, duration = 0;
    switch (type)
    {
    case 0:
        for (size_t i = 0; i < innerCir; i++)
        {
            start = get_time_ns();
            matmul_plain(N, A, B, C);
            end = get_time_ns();
            duration = end - start;
            if (duration < time_min) time_min = duration;
        }
        break;
    case 1:
        for (size_t i = 0; i < innerCir; i++)
        {
            start = get_time_ns();
            matmul_improved(N, A, B, C);
            end = get_time_ns();
            duration = end - start;
            if (duration < time_min) time_min = duration;
        }
        break;
    default:
        for (size_t i = 0; i < innerCir; i++)
        {
            start = get_time_ns();
            matmul_openblas(N, A, B, C);
            end = get_time_ns();
            duration = end - start;
            if (duration < time_min) time_min = duration;
        }
        break;
    }

    return time_min;
}

void test(int size, int cir0, int cir1, int cir2, int outerCir,FILE *fp)
{
    long long r1 = LONG_LONG_MAX, r2 = LONG_LONG_MAX, r3 = LONG_LONG_MAX;
    for (size_t i = 0; i < outerCir; i++)
    {
        struct Matrix *A = create_matrix(size, size);
        struct Matrix *B = create_matrix(size, size);
        r1 = test_inner(size, cir0, 6, A, B);
        Sleep(50);
        r2 = test_inner(size, cir1, 1, A, B);
        Sleep(50);
        r3= test_inner(size, cir2, 2, A, B);
        Sleep(50);
    }

    printf(" N = %d, outer %d, plain %lld, improved %lld, openblas %lld\n" , size, outerCir, r1, r2, r3);
    fprintf(fp, "%d,%lld,%lld,%lld\n", size, r1, r2, r3);

    
}


int main()
{
    // 64K is too large for normal memory allocation
    // check(16, 10000);
    // check(128, 1000);
    // check(400, 100);
    // check(800, 10);
    // check(1024, 3);
    // check(8192, 1);


    FILE *fp = fopen("../out/result.csv", "w");
    if (!fp) {
        printf("Failed to open ../out/result.csv for writing.\n");
        return -1;
    }
    fprintf(fp, "Size,Plain(ns),Improved(ns),OpenBLAS(ns)\n");


    test(16, 10, 100, 100, 10,fp);
    Sleep(1000);

    test(128, 10, 100, 100, 1000,fp);
    Sleep(1000);

    test(256, 1, 10, 10, 1000,fp);
    Sleep(1000);

    test(512, 1, 100, 100, 100,fp);
    Sleep(1000);
    test(1024, 1, 10, 10, 10,fp);
    Sleep(1000);
    test(2048, 1, 10, 10, 10,fp);
    Sleep(1000);

    test(8192, 1, 5, 5, 5,fp);
    Sleep(1000);

    //test(64000, 0, 1, 1, 1);
    fclose(fp);
    printf("\n==> Results successfully saved to ../out/result.csv\n");





    return 0;
}
