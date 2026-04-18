#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "main.h"

// 辅助函数：生成一个完全随机的巨大矩阵文件
int generate_disk_matrix(const char *filename, size_t N)
{
    FILE *f = fopen(filename, "wb");
    if (!f) return -1;
    
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
    if (!f) return -1;
    
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
void read_block_from_disk(FILE *f, struct Matrix *block, size_t N, size_t row_start, size_t col_start)
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
void write_block_to_disk(FILE *f, struct Matrix *block, size_t N, size_t row_start, size_t col_start)
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
void matmul_out_of_core(size_t N, int BLOCK_SIZE, const char *fileA, const char *fileB, const char *fileC)
{
    FILE *fA = fopen(fileA, "rb");
    FILE *fB = fopen(fileB, "rb");
    FILE *fC = fopen(fileC, "r+b"); // r+b 允许读写更新

    if (!fA || !fB || !fC) {
        printf("Failed to open matrix files.\n");
        return;
    }

    // 在内存中仅仅分配 3 个 BLOCK_SIZE * BLOCK_SIZE 的微型矩阵
    // 比如 BLOCK是 1024，这里总共只需要 1024*1024*4*3 = 12 MB 内存！
    struct Matrix blockA = {BLOCK_SIZE, BLOCK_SIZE, (float *)malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(float))};
    struct Matrix blockB = {BLOCK_SIZE, BLOCK_SIZE, (float *)malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(float))};
    struct Matrix blockC = {BLOCK_SIZE, BLOCK_SIZE, (float *)malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(float))};

    // 三层循环，遍历所有的宏块
    for (size_t i = 0; i < N; i += BLOCK_SIZE)
    {
        for (size_t j = 0; j < N; j += BLOCK_SIZE)
        {
            // 核心思路：我们要计算 C[i][j] 这个块。它等于 sum(A[i][k] * B[k][j])
            // 1. 先把原来 C 中该位置的底稿读入内存 (我们要不断累加上去)
            read_block_from_disk(fC, &blockC, N, i, j);

            for (size_t k = 0; k < N; k += BLOCK_SIZE) {
                // 2. 读入 A 的宏块 和 B 的宏块
                read_block_from_disk(fA, &blockA, N, i, k);
                read_block_from_disk(fB, &blockB, N, k, j);

                // 3. 在内存中用你已经写好的超快 improved 算出这个宏部分
                // 注意这里是对 blockC 不断累加，所以底层需要将你原来的 C=A*B 稍做改动变为 C+=A*B 
                // 我们调用改良过的核外专用乘加函数 (需稍调,见下方释义)
                matmul_improved(BLOCK_SIZE, &blockA, &blockB, &blockC); 
            }

            // 4. 这一大块 C 彻底算完了（所有 K 都加完了），写回硬盘！
            write_block_to_disk(fC, &blockC, N, i, j);
            printf("Block C[%zu, %zu] completed & written to disk.\n", i, j);
        }
    }

    free(blockA.data); free(blockB.data); free(blockC.data);
    fclose(fA); fclose(fB); fclose(fC);
}

// 注意: 主函数仅为演示，实际执行 64000 依然要花费几天时间（磁盘I/O太慢） 
// 建议用 N=4096 和 BLOCK=1024 测试核外的代码验证有效性。
int main() {
    size_t N = 4096;      // 测试规模
    int BLOCK_SIZE = 1024; // 哪怕 N 有 64000，内存也只会占用 1024*1024 级别的几兆字节！

    // 1. 在硬盘生成矩阵
    printf("Generating massive disk files...\n");
    generate_disk_matrix("A_disk.bin", N);
    generate_disk_matrix("B_disk.bin", N);
    generate_zero_disk_matrix("C_disk.bin", N); // C 初始化为全0
    
    // 2. 利用区区几 MB 内存演算巨型文件
    printf("Starting Out-of-Core Computation...\n");
    matmul_out_of_core(N, BLOCK_SIZE, "A_disk.bin", "B_disk.bin", "C_disk.bin");
    
    printf("Out-of-Core computation completed!\n");
    return 0;
}