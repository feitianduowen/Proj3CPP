#include "main.h"

int record_plain()
{
    FILE *fp = fopen("../out/plot.csv", "w");
    if (!fp)
    {
        printf("Failed to open plot.csv for writing\n");
        return 1;
    }
    fprintf(fp, "Size,Time(ms)\n");
    fprintf(fp, "%d,%lld\n", 16, test_outer(16, 10, 1000, 4) );
    Sleep(1000);
    fprintf(fp, "%d,%lld\n", 128, test_outer(128, 10, 1000, 4) );
    Sleep(1000);
    fprintf(fp, "%d,%lld\n", 256, test_outer(256, 10, 1000, 4) );
    Sleep(1000);
    fprintf(fp, "%d,%lld\n", 512, test_outer(512, 10, 100, 4) );
    Sleep(1000);
    fprintf(fp, "%d,%lld\n", 1024, test_outer(1024, 10, 100, 4) );
    Sleep(1000);
    fprintf(fp, "%d,%lld\n", 2048, test_outer(2048, 10, 100, 4) );
    Sleep(1000);
    fprintf(fp, "%d,%lld\n", 8192, test_outer(8192, 10, 10, 4) );
    Sleep(1000);
    fclose(fp);
    return 0;
}

int check_compare(size_t N, size_t cir,int type1,int type2,float epsilon)
{
    char *type_name1="";
    char *type_name2="";
    switch (type1) {
        case 0:
            type_name1 = "ikj\0";
            break;
        case 1:
            type_name1 = "improved\0";
            break;
        case 2:
            type_name1 = "tp\0";
            break;
        case 3:
            type_name2 = "openblas\0";
            break;
        default:
            type_name2 = "plain\0";
    }
    switch (type2) {
        case 0:
            type_name2 = "ikj\0";
            break;
        case 1:
            type_name2 = "improved\0";
            break;
        case 2:
            type_name2 = "tp\0";
            break;
        case 3:
            type_name2 = "openblas\0";
            break;
        default:
            type_name2 = "plain\0";
    }
    long long time_1 = LONG_LONG_MAX, time_2 = LONG_LONG_MAX, start = 0, end = 0;
    struct Matrix *A = create_matrix(N, N);
    struct Matrix *C1 = create_matrix(N, N);
    struct Matrix *C2 = create_matrix(N, N);
    clear_matrix(C1);
    clear_matrix(C2);
    int result = 0;
    if (!A || !C1 || !C2)
    {
        printf("Memory allocation failed for check test\n");
        return -1;
    }
    for (size_t i = 0; i < cir; i++)
    {
        randomize_matrix(A);
        switch (type1) {
            case 0:
                start = get_time_ns();
                matmul_ikj(N, A, A, C1);
                end = get_time_ns();
                break;
            case 1:
                start = get_time_ns();
                matmul_improved(N, A, A, C1);
                end = get_time_ns();
                break;
            case 2:
                start = get_time_ns();
                matmul_tp(N, A, A, C1);
                end = get_time_ns();
                break;
            case 3:
                start = get_time_ns();
                matmul_openblas(N, A, A, C1);
                end = get_time_ns();
                break;
            default:
                start = get_time_ns();
                matmul_plain(N, A, A, C1);
                end = get_time_ns();
        }
        if (end - start < time_1)
            time_1 = end - start;

        switch (type2) {
            case 0:
                start = get_time_ns();
                matmul_ikj(N, A, A, C2);
                end = get_time_ns();
                break;
            case 1:
                start = get_time_ns();
                matmul_improved(N, A, A, C2);
                end = get_time_ns();
                break;
            case 2:
                start = get_time_ns();
                matmul_tp(N, A, A, C2);
                end = get_time_ns();
                break;
            case 3:
                start = get_time_ns();
                matmul_openblas(N, A, A, C1);
                end = get_time_ns();
                break;
            default:
                start = get_time_ns();
                matmul_plain(N, A, A, C1);
                end = get_time_ns();
        }
        if (end - start < time_2)
            time_2 = end - start;

        result = compare_matrices(C1, C2, epsilon);
        if (!result)
        {
            printf("Result mismatch for N = %zu\n", N);
            break;
        }
    }

    free_matrix_aligned(A);
    free_matrix_aligned(C1);
    free_matrix_aligned(C2);
    printf("%s: %lld ns, %s: %lld ns\n", type_name1,time_1,type_name2, time_2);
    printf("Check for N = %zu %s\n", N, result ? "PASSED" : "FAILED");
    return result;
}

int check_compare_all(int type1, int type2,float epsilon)
{
    check_compare(16, 10000,type1,type2,epsilon);
    Sleep(1000);
    check_compare(64, 10000,type1,type2,epsilon);
    Sleep(10000);
    check_compare(128, 1000,type1,type2,epsilon);
    Sleep(1000);
    check_compare(256, 500,type1,type2,epsilon);
    Sleep(1000);
    check_compare(400, 1000,type1,type2,epsilon);
    Sleep(1000);
    check_compare(800, 100,type1,type2,epsilon);
    Sleep(1000);
    check_compare(1024, 100,type1,type2,epsilon);
    Sleep(1000);
    check_compare(2048, 5,type1,type2,epsilon);
    Sleep(1000);
    check_compare(4096, 2,type1,type2,epsilon);
    return 0;
}

int check_aligned(){
    long long r16 = test_outer(16, 10, 1000, 4);
    printf(".");
    Sleep(1000);
    long long r64 = test_outer(64, 10, 1000, 4);
    printf(".");
    Sleep(1000);
    long long r128 = test_outer(128, 10, 1000, 4);
    printf(".");
    Sleep(1000);
    long long r256 = test_outer(256, 10, 1000, 4);
    printf(".");
    Sleep(1000);
    long long r512 = test_outer(512, 10, 1000, 4);
    printf(".");
    Sleep(1000);
    long long r1024 = test_outer(1024, 10, 100, 4);
    printf(".");
    Sleep(1000);
    long long r2048 = test_outer(2048, 10, 100, 4);
    printf(".");
    Sleep(1000);
    long long r4096 = test_outer(4096, 10, 10, 4);
    printf(".");
    Sleep(1000);
    long long r8192 = test_outer(8192, 10, 5, 4);

    printf("\nsize,time\n16,%lld\n64,%lld\n128,%lld\n256,%lld\n512,%lld\n1024,%lld\n2048,%lld\n4096,%lld\n8192,%lld",r16,r64,r128,r256,r512,r1024,r2048,r4096,r8192);
    return 0;
}

int main()
{
    // record_plain();
    // all_ikj();
    //check_compare_all(2, 1, 1e-5f);
    check_aligned();

    return 0;
}