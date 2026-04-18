#include "main.h"

int sizes[] = {16, 128, 400, 800, 1024, 8192, 64000};
int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

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
        if ((int)(i*100000/outerCir)%2000 == 0) printf(".");
    }
    printf("\n");

    if (cir0 != 0)
        time_plain /= outerCir;
    else
        time_plain = 0.0;
    time_improved /= outerCir;
    time_openblas /= outerCir;

    if (res)
    {
        res->size = sizeN;
        if (cir0 != 0) res->time_plain = time_plain;
        else res->time_plain = 0.0;
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

        result = compare_matrices(C1, C2, 1e-3f);
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

    test(16, 10, 100, 100, 10000, &results[res_idx++]);
    test(128, 10, 100, 100, 1000, &results[res_idx++]);
    test(400, 1, 10, 10, 1000, &results[res_idx++]);
    test(800, 1, 100, 100, 100, &results[res_idx++]);
    test(1024, 1, 10, 10, 10, &results[res_idx++]);
    test(8192, 1, 5, 5, 5, &results[res_idx++]);
    //test(64000, 0, 1, 1, 1, &results[res_idx++]);

    FILE *fp = fopen("../out/result.csv", "w");
    if (fp)
    {
        fprintf(fp, "Size,Plain(ns),Improved(ns),OpenBLAS(ns)\n");
        for (int i = 0; i < res_idx; i++)
        {
            fprintf(fp, "%d,%.2f,%.2f,%.2f\n", results[i].size, results[i].time_plain, results[i].time_improved, results[i].time_openblas);
        }
        fclose(fp);
        printf("\n==> Results successfully saved to ../out/result.csv\n");
    }
    else
    {
        printf("\n==> Failed to open ../out/result.csv for writing.\n");
    }

    return 0;
}
