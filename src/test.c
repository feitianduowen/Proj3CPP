#include "main.h"

const char *fileA1 = "../out/A_disk_test.bin";
const char *fileB1 = "../out/B_disk_test.bin";
const char *fileC1 = "../out/C_disk_test.bin";

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

Matrix *create_matrix(size_t rows, size_t cols, enum TYPE_NAME type_name)
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
    mat->data = type_name == ALIGNED ? (float *)_aligned_malloc(rows * cols * sizeof(float), 32) : (float *)malloc(len * sizeof(float));
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

void free_matrix(Matrix **mmat, enum TYPE_NAME type_name)
{

    if (mmat == NULL)
        return;
    Matrix *mat = *mmat;
    if (mat != NULL)
    {
        if (mat->data)
        {
            if (type_name == ALIGNED)
            {
                _aligned_free(mat->data);
            }
            else
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

// 辅助函数：将内存中的矩阵写出至文件
int matrix_putin(const char *filename, size_t N, Matrix *m)
{
    if (N > 9000)
    {
        printf("Matrix size N = %zu is too large for in-memory checking.\n", N);
        return -1;
    }
    if (!m || !m->data)
        return -1;
    FILE *f = fopen(filename, "wb");
    if (!f)
        return -1;

    size_t written = fwrite(m->data, sizeof(float), N * N, f);
    fclose(f);
    return (written == N * N) ? 0 : -1;
}

// 辅助函数：从文件读取矩阵至内存
int matrix_readout(const char *filename, size_t N, Matrix *m)
{
    if (N > 9000)
    {
        printf("Matrix size N = %zu is too large for out-of-core checking.\n", N);
        return -1;
    }
    if (!m || !m->data)
        return -1;
    FILE *f = fopen(filename, "rb");
    if (!f)
        return -1;

    size_t read_cnt = fread(m->data, sizeof(float), N * N, f);
    fclose(f);
    return (read_cnt == N * N) ? 0 : -1;
}

int generate_disk_matrix(const char *filename, size_t N)
{
    FILE *f = fopen(filename, "wb");
    if (!f)
        return -1;

    // 为了防止内存爆炸，我们每次只在内存生成一行，然后写入文件
    float *row_buffer = (float *)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < N; j++)
            row_buffer[j] = (float)rand() / RAND_MAX;
        fwrite(row_buffer, sizeof(float), N, f);
    }
    free(row_buffer);
    fclose(f);
    return 0;
}

// 辅助函数：初始化一个全0的巨大矩阵文件
int generate_zero_disk_matrix(const char *filename, size_t N)
{
    FILE *f = fopen(filename, "wb");
    if (!f)
        return -1;

    float *row_buffer = (float *)calloc(N, sizeof(float));
    for (size_t i = 0; i < N; i++)
    {
        fwrite(row_buffer, sizeof(float), N, f);
    }
    free(row_buffer);
    fclose(f);
    return 0;
}

// 验证核外观算法与纯内存标准答案是否一致
int verify_ooc_result(size_t N, const char *fileA, const char *fileB, const char *fileC, const int BLOCK_SIZE, size_t cir)
{
    if (N > 10000)
    {
        printf("N=%zu is too large to comfortably verify in main memory, skipping verification.\n", N);
        return 0;
    }
    float max = 0.0f;
    double sum = 0.0f;
    int length = (int)(N * N);
    size_t cntG = 0;
    size_t cnt41 = 0, cnt42 = 0, cnt43 = 0, cnt44 = 0, cnt45 = 0, cnt46 = 0, cnt47 = 0, cnt48 = 0, cnt49 = 0;
    size_t cnt31 = 0, cnt32 = 0, cnt33 = 0, cnt34 = 0, cnt35 = 0, cnt36 = 0, cnt37 = 0, cnt38 = 0, cnt39 = 0;
    size_t cnt2 = 0;
    for (size_t i = 0; i < cir; i++)
    {
        Matrix *mat_a_test = create_matrix(N, N, ALIGNED);
        Matrix *mat_b_test = create_matrix(N, N, ALIGNED);
        Matrix *mat_c_test = create_matrix(N, N, ALIGNED);
        Matrix *mat_c_check = create_matrix(N, N, ALIGNED);
        if (!mat_a_test || !mat_b_test || !mat_c_test || !mat_c_check)
        {
            printf("Memory allocation failed during verification setup.\n");
            free_matrix(&mat_a_test, ALIGNED);
            free_matrix(&mat_b_test, ALIGNED);
            free_matrix(&mat_c_test, ALIGNED);
            free_matrix(&mat_c_check, ALIGNED);
            return -1;
        }
        randomize_matrix(mat_a_test);
        randomize_matrix(mat_b_test);
        // 用 OpenBLAS 算出正确的标准答案，存放在 mat_c_check 里，不要去覆盖它
        clear_matrix(mat_c_check);
        clear_matrix(mat_c_test);
        type_switch(N,IMPROVED2, mat_a_test, mat_b_test, mat_c_check);

        // printf("Generating massive disk files...\n");
        matrix_putin(fileA, N, mat_a_test);
        matrix_putin(fileB, N, mat_b_test);
        generate_zero_disk_matrix(fileC, N); // C 初始化为全0
\
        matmul_out_of_core(N, BLOCK_SIZE, fileA, fileB, fileC);
        if (matrix_readout(fileC, N, mat_c_test) != 0)
        {
            printf("Failed to read back C_disk.bin, verification may be compromised.\n");
        }

        // mat_c_check: OpenBLAS 算出的纯内存标准答案
        // mat_c_test: OOC 读出来并加载的结果

        for (size_t j = 0; j < length; j++)
        {
            float diff = fabsf(mat_c_check->data[j] - mat_c_test->data[j]);
            sum += diff;
            if (diff == 0)
            {
                cntG++;
            }
            else if (diff < 1e-5f)
            {
                cnt41++;
            }
            else if (diff < 2e-5f)
            {
                cnt42++;
            }
            else if (diff < 3e-5f)
            {
                cnt43++;
            }
            else if (diff < 4e-5f)
            {
                cnt44++;
            }
            else if (diff < 5e-5f)
            {
                cnt45++;
            }
            else if (diff < 6e-5f)
            {
                cnt46++;
            }
            else if (diff < 7e-5f)
            {
                cnt47++;
            }
            else if (diff < 8e-5f)
            {
                cnt48++;
            }
            else if (diff < 9e-5f)
            {
                cnt49++;
            }
            else if (diff < 1e-4f)
            {
                cnt31++;
            }
            else if (diff < 2e-4f)
            {
                cnt32++;
            }
            else if (diff < 3e-4f)
            {
                cnt33++;
            }
            else if (diff < 4e-4f)
            {
                cnt34++;
            }
            else if (diff < 5e-4f)
            {
                cnt35++;
            }
            else if (diff < 6e-4f)
            {
                cnt36++;
            }
            else if (diff < 7e-4f)
            {
                cnt37++;
            }
            else if (diff < 8e-4f)
            {
                cnt38++;
            }
            else if (diff < 9e-4f)
            {
                cnt39++;
            }
            else
                cnt2++;
            if (diff > max)
                max = diff;
        }
        // printf("Cleaning up temporary verification disk files...\n");
        remove(fileA);
        remove(fileB);
        remove(fileC);
        free_matrix(&mat_a_test, ALIGNED);
        free_matrix(&mat_b_test, ALIGNED);
        free_matrix(&mat_c_test, ALIGNED);
        free_matrix(&mat_c_check, ALIGNED);
    }

    printf("OOC N=%zu,BLOCK_SIZE=%d, Max error: %e, Average error: %e\n, cnt0=%zu, 41=%zu, 42=%zu, 43=%zu, 44=%zu, 45=%zu, 46=%zu, 47=%zu, 48=%zu, 49=%zu, 31=%zu, 32=%zu, 33=%zu, 34=%zu, 35=%zu, 36=%zu, 37=%zu, 38=%zu, 39=%zu, 2=%zu.\n", N, BLOCK_SIZE, max, sum / length,
           cntG, cnt41, cnt42, cnt43, cnt44, cnt45, cnt46, cnt47, cnt48, cnt49,
           cnt31, cnt32, cnt33, cnt34, cnt35, cnt36, cnt37, cnt38, cnt39, cnt2);

    // 删除用于检查的临时 disk 文件，清理释放硬盘空间

    return 0;
}

int test_ooc(size_t N, int BLOCK_SIZE, int innerCir, int outerCir)
{
    if (outerCir < 1 || innerCir < 1)
    {
        printf("outerCir and innerCir must be at least 1.\n");
        return -1;
    }
    long long time_MIN = LONG_LONG_MAX, tp = 0;
    size_t cnt = 0, total = outerCir * innerCir;
    printf("N = %zu , blocksize = %d , ", N, BLOCK_SIZE);
    for (size_t i = 0; i < outerCir; i++)
    {
        generate_disk_matrix(fileA1, N);
        generate_zero_disk_matrix(fileC1, N);
        // 这里我们不需要每次都生成 B 矩阵了，因为我们之前的 OOC 实现里 A 和 B 是一样的（为了节省时间和空间），所以我们直接复用 A 的文件作为 B 就好了
        for (size_t j = 0; j < innerCir; j++)
        {
            tp = matmul_out_of_core(N, BLOCK_SIZE, fileA1, fileB1, fileC1);
            if (tp < time_MIN)
                time_MIN = tp;
            if ((int)(++cnt * 1000 / total) % 100 == 0)
                printf(".");
        }
    }

    printf("%lld ns!\n", time_MIN);

    // 3. 验证结果正确性

    // 4. 清理测试文件
    remove(fileA1);
    remove(fileB1);
    remove(fileC1);

    return 0;
}

long long test_inner(size_t n, size_t innerCir, enum TYPE_NAME type, const Matrix *A, const Matrix *B, Matrix const *C)
{
    long long time_res = LONG_LONG_MAX, temporary = 0;
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
    long long res = LONG_LONG_MAX, tp = 0;
    Matrix *A = create_matrix(n, n, type);
    Matrix *B = create_matrix(n, n, type);
    Matrix *C = create_matrix(n, n, type);

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
            free_matrix(&A, type);
            free_matrix(&B, type);
            free_matrix(&C, type);
            return -1;
        }
        if (tp < res)
            res = tp;
    }

    free_matrix(&A, type);
    free_matrix(&B, type);
    free_matrix(&C, type);

    return res;
}

int compare(size_t N, size_t cir, enum TYPE_NAME type1, enum TYPE_NAME type2)
{
    char *type_name1 = type_name(type1);
    char *type_name2 = type_name(type2);

    // Always use aligned memory for checks to prevent crashes if one of the tested types is ALIGNED
    enum TYPE_NAME check_type = (type1 == ALIGNED || type2 == ALIGNED) ? ALIGNED : PLAIN;
    Matrix *A = create_matrix(N, N, check_type);
    Matrix *C1 = create_matrix(N, N, check_type);
    Matrix *C2 = create_matrix(N, N, check_type);

    if (!A || !C1 || !C2)
    {
        printf("Memory allocation failed for check test\n");
        return -1;
    }
    float max = 0.0f;
    double sum = 0.0f;

    size_t cntG = 0;
    size_t cnt41 = 0, cnt42 = 0, cnt43 = 0, cnt44 = 0, cnt45 = 0, cnt46 = 0, cnt47 = 0, cnt48 = 0, cnt49 = 0;
    size_t cnt31 = 0, cnt32 = 0, cnt33 = 0, cnt34 = 0, cnt35 = 0, cnt36 = 0, cnt37 = 0, cnt38 = 0, cnt39 = 0;
    size_t cnt2 = 0;
    int length = (int)(N * N);
    int cir0 = (int)cir;
    for (size_t i = 0; i < cir; i++)
    {
        clear_matrix(C1);
        randomize_matrix(A);
        type_switch(N, type1, A, A, C1);
        clear_matrix(C2);
        type_switch(N, type2, A, A, C2);

        double sum0 = 0.0f;
        for (size_t j = 0; j < length; j++)
        {
            float diff = fabsf(C1->data[j] - C2->data[j]);
            sum += diff;
            if (diff == 0)
            {
                cntG++;
            }
            else if (diff < 1e-5f)
            {
                cnt41++;
            }
            else if (diff < 2e-5f)
            {
                cnt42++;
            }
            else if (diff < 3e-5f)
            {
                cnt43++;
            }
            else if (diff < 4e-5f)
            {
                cnt44++;
            }
            else if (diff < 5e-5f)
            {
                cnt45++;
            }
            else if (diff < 6e-5f)
            {
                cnt46++;
            }
            else if (diff < 7e-5f)
            {
                cnt47++;
            }
            else if (diff < 8e-5f)
            {
                cnt48++;
            }
            else if (diff < 9e-5f)
            {
                cnt49++;
            }
            else if (diff < 1e-4f)
            {
                cnt31++;
            }
            else if (diff < 2e-4f)
            {
                cnt32++;
            }
            else if (diff < 3e-4f)
            {
                cnt33++;
            }
            else if (diff < 4e-4f)
            {
                cnt34++;
            }
            else if (diff < 5e-4f)
            {
                cnt35++;
            }
            else if (diff < 6e-4f)
            {
                cnt36++;
            }
            else if (diff < 7e-4f)
            {
                cnt37++;
            }
            else if (diff < 8e-4f)
            {
                cnt38++;
            }
            else if (diff < 9e-4f)
            {
                cnt39++;
            }
            else
                cnt2++;
            if (diff > max)
                max = diff;
        }

        sum += sum0 / (length);
    }

    free_matrix(&A, check_type);
    free_matrix(&C1, check_type);
    free_matrix(&C2, check_type);

    printf("Check for %s -> %s\t N = %zu, max = %e, mean = %e, cnt0=%zu, 41=%zu, 42=%zu, 43=%zu, 44=%zu, 45=%zu, 46=%zu, 47=%zu, 48=%zu, 49=%zu, 31=%zu, 32=%zu, 33=%zu, 34=%zu, 35=%zu, 36=%zu, 37=%zu, 38=%zu, 39=%zu, 2=%zu.\n",
           type_name1, type_name2, N, max, sum / ((double)cir0 * length),
           cntG, cnt41, cnt42, cnt43, cnt44, cnt45, cnt46, cnt47, cnt48, cnt49,
           cnt31, cnt32, cnt33, cnt34, cnt35, cnt36, cnt37, cnt38, cnt39, cnt2);
    return 0;
}

int check_single(enum TYPE_NAME type)
{
    long long r16 = test_outer(16, 10, 100, type);
    printf("16,%lld\n", r16);
    Sleep(1000);
    long long r64 = test_outer(64, 10, 10, type);
    printf("64,%lld\n", r64);
    Sleep(1000);
    long long r128 = test_outer(128, 10, 10, type);
    printf("128,%lld\n", r128);
    Sleep(1000);
    long long r256 = test_outer(256, 10, 10, type);
    printf("256,%lld\n", r256);
    Sleep(1000);
    long long r512 = test_outer(512, 10, 10, type);
    printf("512,%lld\n", r512);
    Sleep(1000);
    long long r1024 = test_outer(1024, 10, 1, type);
    printf("1024,%lld\n", r1024);
    Sleep(1000);
    long long r2048 = test_outer(2048, 10, 1, type);
    printf("2048,%lld\n", r2048);
    Sleep(1000);
    long long r4096 = test_outer(4096, 10, 1, type);
    printf("4096,%lld\n", r4096);
    Sleep(1000);
    long long r8192 = test_outer(8192, 5, 1, type);
    printf("8192,%lld\n", r8192);
    return 0;
}

int oochhh()
{
    // 缩小一下 N，用来验证核外算法的正确性（方便我们同时放入内存进行对比）

    // verify_ooc_result(1024, fileA1,fileB1, fileC1, 256, 10);
    // verify_ooc_result(1024, fileA1,fileB1, fileC1, 512, 10);
    // verify_ooc_result(2048, fileA1, fileB1, fileC1, 512, 10);
    // test_ooc(1024, 256, 10, 5);
    // test_ooc(2048, 256, 3, 5);
    // test_ooc(4096, 256, 1, 5);
    // test_ooc(8192, 256, 1, 3);
    // test_ooc(16384, 256, 1, 1);

    // test_ooc(1024, 512, 10, 5);
    // test_ooc(2048, 512, 3, 5);
    // test_ooc(4096, 512, 3, 5);
    // test_ooc(8192, 512, 1, 5);
    // test_ooc(16384, 512, 1, 1);
    // test_ooc(16384, 1024, 1, 1);
    // test_ooc(16384, 2048, 3, 3);
    // test_ooc(16384, 4096, 1, 3);
    // test_ooc(65536, 4096, 1, 1);

    // test_ooc(8192, 1024, 1, 10);
    // test_ooc(65536, 8192, 1, 1);

    // test_ooc(65536, 16384, 1, 2);
    // test_ooc(2048, 512, 10, 5);
    // test_ooc(4096, 1024, 3, 5);
    // test_ooc(8192, 2048, 3, 5);

    // test_ooc(1024, 128, 100, 100);
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
    // compare(1024, 10, PLAIN, OPENBLAS);
    // compare(1024, 10, PLAIN, IMPROVED3);
    // long long r16 = test_outer(16384, 3, 3, IMPROVED3);
    // printf("16384,%lld\n", r16);
    // oochhh();
    return 0;
}