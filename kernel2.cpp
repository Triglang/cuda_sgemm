#include <cuda_runtime.h>

#define BLOCK_SIZE 32  // 通常选择16或32，取决于硬件和问题规模

// cache blocking version, without register-level data re-use
__global__ void gemm_kernel2(float *A, float *B, float *C, int n, int m, int k) {
    __shared__ float tileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tileB[BLOCK_SIZE][BLOCK_SIZE];
    
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (k + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        int a_col = t * BLOCK_SIZE + threadIdx.x;
        if (row < n && a_col < k) {
            tileA[threadIdx.y][threadIdx.x] = A[row * k + a_col];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        int b_row = t * BLOCK_SIZE + threadIdx.y;
        if (b_row < k && col < m) {
            tileB[threadIdx.y][threadIdx.x] = B[b_row * m + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        for (int i = 0; i < BLOCK_SIZE; i++) {
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < n && col < m) {
        C[row * m + col] = sum;
    }
}