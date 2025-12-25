#include <cuda_runtime.h>

#define BLOCK_SIZE 32  // 通常选择16或32，取决于硬件和问题规模

// cache blocking version, without register-level data re-use
__global__ void gemm_kernel3(float *A, float *B, float *C, int n, int m, int k) {
    __shared__ float tileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tileB[BLOCK_SIZE][BLOCK_SIZE];

    float C_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    int tile_row = threadIdx.y;
    int tile_col = threadIdx.x * 4;
    
    int row = blockIdx.y * BLOCK_SIZE + tile_row;
    int col = blockIdx.x * BLOCK_SIZE + tile_col;

    for (int t = 0; t < k / BLOCK_SIZE; t++) {
        int a_col = t * BLOCK_SIZE + tile_col;
        tileA[tile_row][tile_col] = A[row * k + a_col];
        tileA[tile_row][tile_col + 1] = A[row * k + a_col + 1];
        tileA[tile_row][tile_col + 2] = A[row * k + a_col + 2];
        tileA[tile_row][tile_col + 3] = A[row * k + a_col + 3];
        
        int b_row = t * BLOCK_SIZE + tile_row;
        tileB[tile_row][tile_col] = B[b_row * m + col];
        tileB[tile_row][tile_col + 1] = B[b_row * m + col + 1];
        tileB[tile_row][tile_col + 2] = B[b_row * m + col + 2];
        tileB[tile_row][tile_col + 3] = B[b_row * m + col + 3];
        
        __syncthreads();
        
        for (int i = 0; i < BLOCK_SIZE; i++) {
            C_reg[0] += tileA[tile_row][i] * tileB[i][tile_col];
            C_reg[1] += tileA[tile_row][i] * tileB[i][tile_col + 1];
            C_reg[2] += tileA[tile_row][i] * tileB[i][tile_col + 2];
            C_reg[3] += tileA[tile_row][i] * tileB[i][tile_col + 3];
        }
        
        __syncthreads();
    }

    {
        int t = k / BLOCK_SIZE;
        int a_col = t * BLOCK_SIZE + tile_col;
        if (row < n && a_col < k) {
            tileA[tile_row][tile_col] = A[row * k + a_col];
        } else {
            tileA[tile_row][tile_col] = 0.0f;
        }

        if (row < n && a_col + 1 < k) {
            tileA[tile_row][tile_col + 1] = A[row * k + a_col + 1];
        } else {
            tileA[tile_row][tile_col + 1] = 0.0f;
        }

        if (row < n && a_col + 2 < k) {
            tileA[tile_row][tile_col + 2] = A[row * k + a_col + 2];
        } else {
            tileA[tile_row][tile_col + 2] = 0.0f;
        }

        if (row < n && a_col + 3 < k) {
            tileA[tile_row][tile_col + 3] = A[row * k + a_col + 3];
        } else {
            tileA[tile_row][tile_col + 3] = 0.0f;
        }
        
        int b_row = t * BLOCK_SIZE + tile_row;
        if (b_row < k && col < m) {
            tileB[tile_row][tile_col] = B[b_row * m + col];
        } else {
            tileB[tile_row][tile_col] = 0.0f;
        }

        if (b_row < k && col + 1 < m) {
            tileB[tile_row][tile_col + 1] = B[b_row * m + col + 1];
        } else {
            tileB[tile_row][tile_col + 1] = 0.0f;
        }

        if (b_row < k && col + 2 < m) {
            tileB[tile_row][tile_col + 2] = B[b_row * m + col + 2];
        } else {
            tileB[tile_row][tile_col + 2] = 0.0f;
        }

        if (b_row < k && col + 3 < m) {
            tileB[tile_row][tile_col + 3] = B[b_row * m + col + 3];
        } else {
            tileB[tile_row][tile_col + 3] = 0.0f;
        }
        
        __syncthreads();
        
        for (int i = 0; i < BLOCK_SIZE; i++) {
            C_reg[0] += tileA[tile_row][i] * tileB[i][tile_col];
            C_reg[1] += tileA[tile_row][i] * tileB[i][tile_col + 1];
            C_reg[2] += tileA[tile_row][i] * tileB[i][tile_col + 2];
            C_reg[3] += tileA[tile_row][i] * tileB[i][tile_col + 3];
        }
        
        __syncthreads();
    }
    
    if (row < n && col < m) {
        C[row * m + col] = C_reg[0];
        C[row * m + col + 1] = C_reg[1];
        C[row * m + col + 2] = C_reg[2];
        C[row * m + col + 3] = C_reg[3];
    }
}