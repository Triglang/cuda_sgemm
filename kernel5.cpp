#include <cuda_runtime.h>

#define NS 64
#define MS 64
#define KS 16
#define BLOCK_SIZE 16
#define TILE_SIZE 64

// cache blocking version, without register-level data re-use
__global__ void gemm_kernel5(float *A, float *B, float *C, int n, int m, int k) {
    __shared__ float tileA[NS][KS];
    __shared__ float tileB[KS][MS];

    float4 Av, Bv, C_reg[4];
    float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    C_reg[0] = zero;
    C_reg[1] = zero;
    C_reg[2] = zero;
    C_reg[3] = zero;

    for (int t = 0; t < (k + KS - 1) / KS; t++)
    {  
        int x = threadIdx.x % 4;
        int y_offset = threadIdx.x / 4;
        int y = threadIdx.y * 4;

        int A_row = blockIdx.y * TILE_SIZE + y + y_offset;
        int A_col = t * KS + x * 4;

        bool valid_A_col0 = (A_row < n && A_col < k);
        bool valid_A_col1 = (A_row < n && A_col + 1 < k);
        bool valid_A_col2 = (A_row < n && A_col + 2 < k);
        bool valid_A_col3 = (A_row < n && A_col + 3 < k);

        // tileA[y + y_offset][x * 4] = valid_A_col0 ? A[A_row * k + A_col] : 0.0f;
        // tileA[y + y_offset][x * 4 + 1] = valid_A_col1 ? A[A_row * k + A_col + 1] : 0.0f;
        // tileA[y + y_offset][x * 4 + 2] = valid_A_col2 ? A[A_row * k + A_col + 2] : 0.0f;
        // tileA[y + y_offset][x * 4 + 3] = valid_A_col3 ? A[A_row * k + A_col + 3] : 0.0f;
        tileA[y + y_offset][x * 4] = valid_A_col0 ? A[A_row * k + A_col] : 0.0f;
        tileA[y + y_offset][x * 4 + 1] = valid_A_col1 ? A[A_row * k + A_col + 1] : 0.0f;
        tileA[y + y_offset][x * 4 + 2] = valid_A_col2 ? A[A_row * k + A_col + 2] : 0.0f;
        tileA[y + y_offset][x * 4 + 3] = valid_A_col3 ? A[A_row * k + A_col + 3] : 0.0f;
        
        int B_row = t * KS + threadIdx.y;
        int B_col = blockIdx.x * TILE_SIZE + threadIdx.x * 4;

        Bv = *((float4*)(&B[B_row * m + B_col]));
        *(float4*)(tileB[threadIdx.y] + threadIdx.x * 4) = Bv;

        // bool valid_B0 = (B_row < k && B_col < m);
        // bool valid_B1 = (B_row < k && B_col + 1 < m);
        // bool valid_B2 = (B_row < k && B_col + 2 < m);
        // bool valid_B3 = (B_row < k && B_col + 3 < m);

        // Bv = valid_B0 ? *((float4*)(&B[B_row * m + B_col])) : zero;
        // Bv = valid_B1 ? *((float4*)(&B[B_row * m + B_col + 1])) : zero;
        // Bv = valid_B2 ? *((float4*)(&B[B_row * m + B_col + 2])) : zero;
        // Bv = valid_B3 ? *((float4*)(&B[B_row * m + B_col + 3])) : zero;

        // tileB[threadIdx.y][threadIdx.x * 4] = valid_B0 ? B[B_row * m + B_col] : 0.0f;
        // tileB[threadIdx.y][threadIdx.x * 4 + 1] = valid_B1 ? B[B_row * m + B_col + 1] : 0.0f;
        // tileB[threadIdx.y][threadIdx.x * 4 + 2] = valid_B2 ? B[B_row * m + B_col + 2] : 0.0f;
        // tileB[threadIdx.y][threadIdx.x * 4 + 3] = valid_B3 ? B[B_row * m + B_col + 3] : 0.0f;
        
        __syncthreads();
        
        for (int i = 0; i < KS; i++) {
            C_reg[0].x += tileA[threadIdx.y * 4][i] * tileB[i][threadIdx.x * 4];
            C_reg[0].y += tileA[threadIdx.y * 4][i] * tileB[i][threadIdx.x * 4 + 1];
            C_reg[0].z += tileA[threadIdx.y * 4][i] * tileB[i][threadIdx.x * 4 + 2];
            C_reg[0].w += tileA[threadIdx.y * 4][i] * tileB[i][threadIdx.x * 4 + 3];

            C_reg[1].x += tileA[threadIdx.y * 4 + 1][i] * tileB[i][threadIdx.x * 4];
            C_reg[1].y += tileA[threadIdx.y * 4 + 1][i] * tileB[i][threadIdx.x * 4 + 1];
            C_reg[1].z += tileA[threadIdx.y * 4 + 1][i] * tileB[i][threadIdx.x * 4 + 2];
            C_reg[1].w += tileA[threadIdx.y * 4 + 1][i] * tileB[i][threadIdx.x * 4 + 3];

            C_reg[2].x += tileA[threadIdx.y * 4 + 2][i] * tileB[i][threadIdx.x * 4];
            C_reg[2].y += tileA[threadIdx.y * 4 + 2][i] * tileB[i][threadIdx.x * 4 + 1];
            C_reg[2].z += tileA[threadIdx.y * 4 + 2][i] * tileB[i][threadIdx.x * 4 + 2];
            C_reg[2].w += tileA[threadIdx.y * 4 + 2][i] * tileB[i][threadIdx.x * 4 + 3];

            C_reg[3].x += tileA[threadIdx.y * 4 + 3][i] * tileB[i][threadIdx.x * 4];
            C_reg[3].y += tileA[threadIdx.y * 4 + 3][i] * tileB[i][threadIdx.x * 4 + 1];
            C_reg[3].z += tileA[threadIdx.y * 4 + 3][i] * tileB[i][threadIdx.x * 4 + 2];
            C_reg[3].w += tileA[threadIdx.y * 4 + 3][i] * tileB[i][threadIdx.x * 4 + 3];
        }
        
        __syncthreads();
    }
    
    int C_row = blockIdx.y * TILE_SIZE + threadIdx.y * 4;
    int C_col = blockIdx.x * TILE_SIZE + threadIdx.x * 4;

    if (C_row < n && C_col < m) {
        *((float4*)(&C[C_row * m + C_col])) = C_reg[0];
        *((float4*)(&C[(C_row + 1) * m + C_col])) = C_reg[1];
        *((float4*)(&C[(C_row + 2) * m + C_col])) = C_reg[2];
        *((float4*)(&C[(C_row + 3) * m + C_col])) = C_reg[3];
    }
}