#include "main.h"

char *type_name(enum TYPE_NAME type)
{
    switch (type)
    {
    case IKJ:
        return "ikj";
    case IMPROVED:
        return "improved";
    case TP:
        return "tp";
    case OPENBLAS:
        return "openblas";
    case ALIGNED:
        return "aligned";
    case STRASSEN:
        return "strassen";
    case IMPROVED2:
        return "improved2";
    case IMPROVED3:
        return "improved3";
    case PLAIN:
        return "plain";
    default:
        return "unknown";
    }
}

Matrix *create_matrix(size_t rows, size_t cols)
{
    if (rows > 20000 || cols > 20000)
    {
        printf("Matrix size N = %zu is too large for in-memory handling.\n", rows);
        return NULL;
    }

    size_t len = rows * cols;
    if (len == 0)
    {
        fprintf(stderr, "Invalid size. The input should be > 0.\n");
        return NULL;
    }

    Matrix *mat = (Matrix *)malloc(sizeof(Matrix));

    if (!mat)
    {
        fprintf(stderr, "Allocate host memory failed.\n");
        goto ERR_TAG;
    }
    mat->rows = rows;
    mat->cols = cols;

        mat->data = (float *)malloc(len * sizeof(float));
    

    if (!mat->data)
    {
        fprintf(stderr, "Allocate host memory failed.\n");
        goto ERR_TAG;
    }
    return mat;
ERR_TAG:
    if (mat && mat->data)
        free(mat->data);
    if (mat)
        free(mat);
    return NULL;
}

void free_matrix(Matrix **mmat)
{

    if (mmat == NULL)
        return;
    Matrix *mat = *mmat;
    if (mat != NULL)
    {
        if (mat->data)
        {
            free(mat->data);
        }
    }
    *mmat = NULL;
}

void randomize_matrix(Matrix const *mat)
{
    if (!mat || !mat->data)
        return;
    size_t count = mat->rows * mat->cols;
    for (size_t i = 0; i < count; i++)
        mat->data[i] = (float)rand() / RAND_MAX;
}

long long test_inner(size_t n, size_t innerCir, enum TYPE_NAME type, const Matrix *A, const Matrix *B, Matrix const *C)
{
    long long time_res = LLONG_MAX, temporary = 0;
    if (!A || !B || !C)
    {
        printf("Memory allocation failed for N = %zu\n", n);
        return -1;
    }

    for (size_t j = 0; j < innerCir; j++)
    {
        temporary = type_switch(n, type, A, B, C);
        if (temporary < time_res)
            time_res = temporary;
    }
    return time_res;
}

long long test_outer(size_t n, size_t innerCir, size_t outerCir, enum TYPE_NAME type)
{
    long long res = LLONG_MAX, tp = 0;
    Matrix *A = create_matrix(n, n);
    Matrix *B = create_matrix(n, n);
    Matrix *C = create_matrix(n, n);

    for (size_t i = 0; i < outerCir; i++)
    {
        clear_matrix(A);
        clear_matrix(B);
        randomize_matrix(A);
        randomize_matrix(B);
        clear_matrix(C);
        tp = test_inner(n, innerCir, type, A, B, C);

        if (tp == -1)
        {
            printf("Test failed for N = %zu in outer loop %zu\n", n, i);
            free_matrix(&A);
            free_matrix(&B);
            free_matrix(&C);
            return -1;
        }
        if (tp < res)
            res = tp;
    }

    free_matrix(&A);
    free_matrix(&B);
    free_matrix(&C);

    return res;
}


int check_single(enum TYPE_NAME type)
{
    // long long r16 = test_outer(16, 10, 100, type);
    // printf("16,%lld\n", r16);
    // sleep(1);
    // long long r64 = test_outer(64, 10, 100, type);
    // printf("64,%lld\n", r64);
    // sleep(1);
    // long long r128 = test_outer(128, 10, 100, type);
    // printf("128,%lld\n", r128);
    // sleep(1);
    // long long r256 = test_outer(256, 10, 10, type);
    // printf("256,%lld\n", r256);
    // sleep(1);
    // long long r512 = test_outer(512, 10, 10, type);
    // printf("512,%lld\n", r512);
    // sleep(1);
    // long long r1024 = test_outer(1024, 10, 1, type);
    // printf("1024,%lld\n", r1024);
    // sleep(1);
    // long long r2048 = test_outer(2048, 10, 1, type);
    // printf("2048,%lld\n", r2048);
    // sleep(1);
    // long long r4096 = test_outer(4096, 10, 1, type);
    // printf("4096,%lld\n", r4096);
    // sleep(1);
    long long r8192 = test_outer(8192, 5, 5, type);
    printf("8192,%lld\n", r8192);
    return 0;
}

int main()
{
    // record_plain();
    // all_ikj();
    // check_compare_ser( ALIGNED,IMPROVED, 1e-5f);
    // check_ii(IMPROVED);
    // check_single(ALIGNED);
    // check_single(IKJ);
    // printf("%lld",test_outer(64, 10, 10, PLAIN));
    // // check_single(IMPROVED);
    // // check_compare(1024,3, PLAIN,IKJ, 0);
    // compare(1024, 10, PLAIN, STRASSEN);
    // compare(1024, 10, PLAIN, IMPROVED);
    // compare(1024, 10, PLAIN, ALIGNED);
    // compare(1024, 10, PLAIN, OPENBLAS);
    // compare(1024, 1, PLAIN, PLAIN);
    // check_single(OPENBLAS);
    check_single(IMPROVED3);
    // oochhh();
    return 0;
}