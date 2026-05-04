//call "D:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
//nvcc matmul.cu -o matmul.exe -Xcompiler "/openmp /arch:AVX2 /utf-8"
//nvcc matmul.cu -o matmul.exe -allow-unsupported-compiler
//nvcc CUDA/matmul.cu -o cuda_matmul.exe -allow-unsupported-compiler -I./OpenBLAS -L./OpenBLAS -lopenblas
#include <stdio.h>
#include <time.h>
#include <windows.h>
#include <math.h>
#include <stddef.h>
#include <immintrin.h>
#include <limits.h>
#define BLOCK_SIZE 16
long long get_time_ns()
{
    struct timespec ts;
    // C11 标准时间函数，能获取纳秒精度（在 GCC/MinGW 下完美支持）
    timespec_get(&ts, TIME_UTC);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}
typedef struct 
{
    size_t rows;
    size_t cols;
    float * data; // CPU memory
    float * data_device; //GPU mememory
} Matrix;

Matrix * createMatrix(size_t r, size_t c)
{
    size_t len = r * c;
    if(len == 0)
    {
        fprintf(stderr, "Invalid size. The input should be > 0.\n");
        return NULL;
    }
    Matrix * p  = (Matrix *) malloc(sizeof(Matrix));
    if (p == NULL)
    {
        fprintf(stderr, "Allocate host memory failed.\n");
        goto ERR_TAG;
    }
    p->rows = r;
    p->cols = c;
    p->data = (float*)malloc(sizeof(float)*len);
    if(p->data == NULL)
    {
        fprintf(stderr, "Allocate host memory failed.\n");
        goto ERR_TAG;
    }
    if (cudaMalloc (&p->data_device, sizeof(float) * len) != cudaSuccess)
    {
        fprintf(stderr, "Allocate device memory failed.\n");
        goto ERR_TAG;
    }
    return p;
  ERR_TAG:
    if(p && p->data) free(p->data);
    if(p) free(p);
    return NULL;
}

void freeMatrix(Matrix ** pp)
{
    if(pp == NULL) return;
    Matrix * p = *pp;
    if(p != NULL)
    {
        if(p->data) free(p->data);
        if(p->data_device) cudaFree(p->data_device);
    }
    *pp = NULL;
}
// a simple function to set all elements to the same value
bool setMatrix(Matrix *const pMat, float val)
{
    if(pMat == NULL)
    {
        fprintf(stderr, "NULL pointer.\n");
        return false;
    }
    size_t len = pMat->rows * pMat->cols;
    for(size_t i = 0; i < len; i++)
        pMat->data[i] = val;

    return true;
}

__global__ void mulKernel(const float * A, const float * B, float * C, int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

bool mulGPU(const Matrix * pMat1, const Matrix * pMat2, Matrix * pMat3)
{
    if( pMat1 == NULL || pMat2 == NULL || pMat3 == NULL)
    {
        fprintf(stderr, "Null pointer.\n");
        return false;
    }
    if (pMat1->cols != pMat2->rows || pMat1->rows != pMat3->rows || pMat2->cols != pMat3->cols)
    {
        fprintf(stderr, "The 3 matrics are not in a valid size for multiplication.\n");
        return false;
    }

    cudaError_t ecode = cudaSuccess;
    int M = pMat1->rows;
    int K = pMat1->cols;
    int N = pMat2->cols;

    cudaMemcpy(pMat1->data_device, pMat1->data, sizeof(float)*M*K, cudaMemcpyHostToDevice);
    cudaMemcpy(pMat2->data_device, pMat2->data, sizeof(float)*K*N, cudaMemcpyHostToDevice);
    
    // 使用 2D 线程块分布 (16x16=256 threads/block)
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    
    mulKernel<<<grid, block>>>(pMat1->data_device, pMat2->data_device, pMat3->data_device, M, N, K);
    
    if ((ecode = cudaGetLastError()) != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(ecode));
        return false;
    }
    cudaMemcpy(pMat3->data, pMat3->data_device, sizeof(float)*M*N, cudaMemcpyDeviceToHost);

    return true;
}



__global__ void mulKernelOpt(const float* A, const float* B, float* C,
                                   int M, int N, int K)
{
    // 共享内存 tile
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // 该线程负责的 C 元素坐标
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float sum = 0.0f;

    // 沿 K 维度循环，每次处理一个大小为 BLOCK_SIZE 的 tile
    for (int kBlock = 0; kBlock < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++kBlock) {

        // 合作加载 A 的 tile 到共享内存
        int aRow = row;
        int aCol = kBlock * BLOCK_SIZE + threadIdx.x;
        if (aRow < M && aCol < K)
            As[threadIdx.y][threadIdx.x] = A[aRow * K + aCol];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // 合作加载 B 的 tile 到共享内存
        int bRow = kBlock * BLOCK_SIZE + threadIdx.y;
        int bCol = col;
        if (bRow < K && bCol < N)
            Bs[threadIdx.y][threadIdx.x] = B[bRow * N + bCol];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();   // 确保整个 tile 都已载入

        // 计算该 tile 对元素的部分乘加
        for (int e = 0; e < BLOCK_SIZE; ++e) {
            sum += As[threadIdx.y][e] * Bs[e][threadIdx.x];
        }

        __syncthreads();   // 避免下一轮加载覆盖当前正在使用的数据
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
bool mulGPU1(const Matrix * pMat1, const Matrix * pMat2, Matrix * pMat3)
{
    if( pMat1 == NULL || pMat2 == NULL || pMat3 == NULL)
    {
        fprintf(stderr, "Null pointer.\n");
        return false;
    }
    if (pMat1->cols != pMat2->rows || pMat1->rows != pMat3->rows || pMat2->cols != pMat3->cols)
    {
        fprintf(stderr, "The 3 matrics are not in a valid size for multiplication.\n");
        return false;
    }

    cudaError_t ecode = cudaSuccess;
    int M = pMat1->rows;
    int K = pMat1->cols;
    int N = pMat2->cols;

    cudaMemcpy(pMat1->data_device, pMat1->data, sizeof(float)*M*K, cudaMemcpyHostToDevice);
    cudaMemcpy(pMat2->data_device, pMat2->data, sizeof(float)*K*N, cudaMemcpyHostToDevice);

    // 使用 2D 线程块分布 (16x16=256 threads/block)
    dim3 block(16, 16);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
          (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    mulKernelOpt<<<grid, block>>>(pMat1->data_device, pMat2->data_device, pMat3->data_device, M, N, K);

    if ((ecode = cudaGetLastError()) != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(ecode));
        return false;
    }
    cudaMemcpy(pMat3->data, pMat3->data_device, sizeof(float)*M*N, cudaMemcpyDeviceToHost);

    return true;
}

int test(size_t N, size_t cir){
    Matrix * pMat1 = createMatrix(N, N);
    Matrix * pMat3 = createMatrix(N, N);
    setMatrix(pMat1, 1.1f);
    long long t1=LLONG_MAX, t2=LLONG_MAX,start=0, end=0;
    printf("Testing matrix multiplication with N = %zu...\n", N);
    for (size_t i = 0; i < cir; i++)
    {
        start=get_time_ns();
        mulGPU(pMat1, pMat1, pMat3);
        end=get_time_ns();
        t1 = (end - start) < t1 ? (end - start) : t1;
        setMatrix(pMat3, 0.0f);
    }

    for (size_t i = 0; i < cir; i++)
    {
        start=get_time_ns();
        mulGPU1(pMat1, pMat1, pMat3);
        end=get_time_ns();
        t2 = (end - start) < t2 ? (end - start) : t2;

        setMatrix(pMat3, 0.0f);
    }
    printf("  GPU = %lld ns.\n", t1);
    printf("  GPUOpt = %lld ns.\n", t2);

    freeMatrix(&pMat1);
    freeMatrix(&pMat3);

    return 0;
}

int main()
{


    int dev_count = 0;
    int dev_id = 0;
    cudaGetDeviceCount(&dev_count);
    cudaSetDevice(0);
    cudaGetDevice(&dev_id);
    printf("You have %d cuda devices.\n", dev_count);
    printf("You are using device %d.\n", dev_id);
    // test(16, 1000);
    // test(64, 1000);
    test(128, 100);
    test(256, 100);
    test(512, 100);
    test(1024, 100);
    test(2048, 50);
    test(4096, 10);
    test(8192, 5);
    test(16384, 1);
    return 0;
}