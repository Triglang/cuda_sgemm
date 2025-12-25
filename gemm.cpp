#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "kernel.h"

// 内存对齐分配 (64字节对齐适配AVX-512)
void *my_malloc(uint64_t size)
{
    void *ptr;
    const size_t alignment = 64;
    if (posix_memalign(&ptr, alignment, size) != 0)
        return nullptr;
    return ptr;
}

void my_gemm(int n, int m, int k, float *d_A, float *d_B, float *d_C)
{
#if 0
    // naive
    dim3 block(32, 32);
    dim3 grid((m + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    gemm_kernel1<<<grid, block>>>(d_A, d_B, d_C, n, m, k);
#elif 0
    // cache blocking
    constexpr int BLOCK_SIZE = 32;

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((m + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    
    gemm_kernel2<<<grid, block>>>(d_A, d_B, d_C, n, m, k);
#elif 0
    // cache blocking with 1 x 4 micro kernel
    constexpr int BLOCK_SIZE = 32;

    dim3 block(BLOCK_SIZE / 4, BLOCK_SIZE);
    dim3 grid((m + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    gemm_kernel3<<<grid, block>>>(d_A, d_B, d_C, n, m, k);
#elif 0
    // cache blocking, using 1 x 4 micro kernel with float4
    constexpr int BLOCK_SIZE = 32;

    dim3 block(BLOCK_SIZE / 4, BLOCK_SIZE);
    dim3 grid((m + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    gemm_kernel4<<<grid, block>>>(d_A, d_B, d_C, n, m, k);
#elif 1
    // cache blocking, using 4 x 4 micro kernel with float4 under 256 threads per block
    constexpr int BLOCK_SIZE = 16;
    constexpr int TILE_SIZE = BLOCK_SIZE * 4;
    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);  // (16, 16)
    dim3 grid((m + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

    gemm_kernel5<<<grid, block>>>(d_A, d_B, d_C, n, m, k);
#endif
}