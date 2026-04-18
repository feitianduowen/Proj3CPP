#include "main.h"

const char *fileA1 = "../out/A_disk_test.bin";
const char *fileC1 = "../out/C_disk_test.bin";
// 辅助函数：将内存中的矩阵写出至文件
int matrix_putin(const char *filename, size_t N, struct Matrix *m)
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
int matrix_readout(const char *filename, size_t N, struct Matrix *m)
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

// 辅助函数：从硬盘的 N*N 矩阵中，读取一个左上角开头为 (row_start, col_start)，大小为 BLOCK*BLOCK 的块
void read_block_from_disk(FILE *f, struct Matrix const *block, size_t N, size_t row_start, size_t col_start)
{
    size_t B = block->rows; // BLOCK_SIZE
    for (size_t i = 0; i < B; i++)
    {
        // 计算在巨型矩阵中对应的绝对偏移量 (字节)
        size_t offset = ((row_start + i) * N + col_start) * sizeof(float);
        _fseeki64(f, offset, SEEK_SET); // Windows 下支持超大文件偏移的 fseek
        fread(&block->data[i * B], sizeof(float), B, f);
    }
}

// 辅助函数：将内存中的块写回硬盘的对应位置
void write_block_to_disk(FILE *f, struct Matrix const*block, size_t N, size_t row_start, size_t col_start)
{
    size_t B = block->rows; // BLOCK_SIZE
    for (size_t i = 0; i < B; i++)
    {
        size_t offset = ((row_start + i) * N + col_start) * sizeof(float);
        _fseeki64(f, offset, SEEK_SET);
        fwrite(&block->data[i * B], sizeof(float), B, f);
    }
}

// 核心：核外矩阵乘法
long long matmul_out_of_core(size_t N, int BLOCK_SIZE, const char *fileA, const char *fileB, const char *fileC)
{
    FILE *fA = fopen(fileA, "rb");
    FILE *fB = fopen(fileB, "rb");
    FILE *fC = fopen(fileC, "r+b"); // r+b 允许读写更新

    if (!fA || !fB || !fC)
    {
        printf("Failed to open matrix files.\n");
        return -1;
    }

    // 在内存中仅仅分配 3 个 BLOCK_SIZE * BLOCK_SIZE 的微型矩阵
    struct Matrix blockA = {BLOCK_SIZE, BLOCK_SIZE, (float *)malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(float))};
    struct Matrix blockB = {BLOCK_SIZE, BLOCK_SIZE, (float *)malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(float))};
    struct Matrix blockC = {BLOCK_SIZE, BLOCK_SIZE, (float *)malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(float))};

    long long start_time = get_time_ns();
    // 三层循环，遍历所有的宏块
    for (size_t i = 0; i < N; i += BLOCK_SIZE)
    {
        for (size_t j = 0; j < N; j += BLOCK_SIZE)
        {
            // 1. 先把原来 C 中该位置的底稿读入内存 (由于我们在 main 里面初始化了 C 为 0，第一次读进来的全是 0)
            read_block_from_disk(fC, &blockC, N, i, j);

            for (size_t k = 0; k < N; k += BLOCK_SIZE)
            {
                // 2. 读入 A 的宏块 和 B 的宏块
                read_block_from_disk(fA, &blockA, N, i, k);
                read_block_from_disk(fB, &blockB, N, k, j);

                // 3. 在内存中直接累加本次计算到 blockC
                // 由于 improved 本身的底层逻辑就是 c0 = _mm256_add_ps(c0, ...)
                // 它天然就会执行 C += A * B 的累加，因此不需要中转矩阵了！
                matmul_openblas_accumulate(BLOCK_SIZE, &blockA, &blockB, &blockC);
            }

            // 4. 这一大块 C 彻底算完了（所有 K 都加完了），写回硬盘！
            write_block_to_disk(fC, &blockC, N, i, j);
        }
        //printf("Out-of-Core Row %zu / %zu completed.\n", i, N);
    }
    long long end_time = get_time_ns();

    free(blockA.data);
    free(blockB.data);
    free(blockC.data);
    fclose(fA);
    fclose(fB);
    fclose(fC);
    return end_time - start_time;
}

// 验证核外观算法与纯内存标准答案是否一致
int verify_ooc_result(size_t N, const char *fileA, const char *fileC, const int BLOCK_SIZE, float epsilon)
{
    if (N > 2048)
    {
        printf("N=%zu is too large to comfortably verify in main memory, skipping verification.\n", N);
        return 0;
    }
    struct Matrix *mat_a_test = create_matrix(N, N);
    struct Matrix *mat_c_test = create_matrix(N, N);
    struct Matrix *mat_c_check = create_matrix(N, N);
    randomize_matrix(mat_a_test);

    // 用 OpenBLAS 算出正确的标准答案，存放在 mat_c_check 里，不要去覆盖它
    clear_matrix(mat_c_check);
    matmul_openblas(N, mat_a_test, mat_a_test, mat_c_check);

    printf("Generating massive disk files...\n");
    matrix_putin(fileA, N, mat_a_test);
    
    generate_zero_disk_matrix(fileC, N); // C 初始化为全0

    // 2. 利用区区几 MB 内存演算巨型文件
    printf("checking Out-of-Core Computation...\n");
    long long start_time = get_time_ns();
    matmul_out_of_core(N, BLOCK_SIZE, fileA, fileA, fileC);
    long long end_time = get_time_ns();
    printf("Out-of-Core checking completed! %lld ns\n", end_time - start_time);

    // 3. 开启验证
    // 将 OOC 算出来的结果，读回给 mat_c_test 进行比对！(这也就是您之前命名混了的地方)
    if (matrix_readout(fileC, N, mat_c_test) == 0)
    {
        printf("Successfully read back C_disk.bin for verification.\n");
    }
    else
    {
        printf("Failed to read back C_disk.bin, verification may be compromised.\n");
    }

    // mat_c_check: OpenBLAS 算出的纯内存标准答案
    // mat_c_test: OOC 读出来并加载的结果
    if (compare_matrices(mat_c_check, mat_c_test, epsilon))
    {
        printf("Verification PASSED!");
    }
    else
    {
        printf("Verification FAILED! OOC result diverges significantly.");
        return -1;
    }

    // 删除用于检查的临时 disk 文件，清理释放硬盘空间
    printf("Cleaning up temporary verification disk files...\n");
    remove(fileA);
    remove(fileC);
    free(mat_a_test);
    free(mat_c_test);
    free(mat_c_check);
    return 0;
}

int test_ooc(size_t N, int BLOCK_SIZE, int innerCir, int outerCir)
{
    if (outerCir < 1 || innerCir < 1)
    {
        printf("outerCir and innerCir must be at least 1.\n");
        return -1;
    }
    long long time_MIN = LONG_LONG_MAX, tp=0;
    size_t cnt=0, total = outerCir*innerCir;
    printf("N = %zu , blocksize = %d , ", N, BLOCK_SIZE);
    for (size_t i = 0; i < outerCir; i++)
    {
        generate_disk_matrix(fileA1, N);
        generate_zero_disk_matrix(fileC1, N);
        // 这里我们不需要每次都生成 B 矩阵了，因为我们之前的 OOC 实现里 A 和 B 是一样的（为了节省时间和空间），所以我们直接复用 A 的文件作为 B 就好了
        for (size_t j = 0; j < innerCir; j++)
        {
            tp = matmul_out_of_core(N, BLOCK_SIZE, fileA1, fileA1, fileC1);
            if (tp < time_MIN)time_MIN = tp;
            if ((int)(++cnt * 1000/ total) % 100 == 0)
            printf(".");
        }
    }

    printf("%lld ns!\n", time_MIN);

    // 3. 验证结果正确性

    // 4. 清理测试文件
    remove(fileA1);
    remove(fileC1);

    return 0;
}

int main()
{
    // 缩小一下 N，用来验证核外算法的正确性（方便我们同时放入内存进行对比）

    if (verify_ooc_result(1024, fileA1, fileC1, 256,2e-4f)==-1)return -1;
    if (verify_ooc_result(2048, fileA1, fileC1, 512,4e-4f)==-1)return -1;
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

    test_ooc(1024,128,100,100);
    return 0;
}