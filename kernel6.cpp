#include <cuda_runtime.h>

#define NS 128
#define MS 128
#define KS 8
#define TILE_SIZE 128

// cache blocking version, without register-level data re-use
__global__ void gemm_kernel6(float *A, float *B, float *C, int n, int m, int k) {
    __shared__ float tileA[NS][KS];
    __shared__ float tileB[KS][MS];

    float4 Av, Bv, C_reg[8][2];
    float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int i = 0; i < 8; i++) {
        C_reg[i][0] = zero;
        C_reg[i][1] = zero;
    }

    for (int t = 0; t < (k + KS - 1) / KS; t++)
    {  
        int x = threadIdx.x % 2;
        int y_offset = threadIdx.x / 2;
        int y = threadIdx.y * 8;

        int A_row = blockIdx.y * TILE_SIZE + y + y_offset;
        int A_col = t * KS + x * 4;

        bool valid_A_col0 = (A_row < n && A_col < k);
        bool valid_A_col1 = (A_row < n && A_col + 1 < k);
        bool valid_A_col2 = (A_row < n && A_col + 2 < k);
        bool valid_A_col3 = (A_row < n && A_col + 3 < k);

        tileA[y + y_offset][x * 4] = valid_A_col0 ? A[A_row * k + A_col] : 0.0f;
        tileA[y + y_offset][x * 4 + 1] = valid_A_col1 ? A[A_row * k + A_col + 1] : 0.0f;
        tileA[y + y_offset][x * 4 + 2] = valid_A_col2 ? A[A_row * k + A_col + 2] : 0.0f;
        tileA[y + y_offset][x * 4 + 3] = valid_A_col3 ? A[A_row * k + A_col + 3] : 0.0f;

        int x_offset = threadIdx.y % 2;
        int Bx = x_offset * 64 + threadIdx.x * 4;
        int By = threadIdx.y / 2;
        
        int B_row = t * KS + By;
        int B_col = blockIdx.x * TILE_SIZE + Bx;

        Bv = *((float4*)(&B[B_row * m + B_col]));
        *(float4*)(tileB[By] + Bx) = Bv;

        // bool valid_B0 = (B_row < k && B_col < m);
        // bool valid_B1 = (B_row < k && B_col + 1 < m);
        // bool valid_B2 = (B_row < k && B_col + 2 < m);
        // bool valid_B3 = (B_row < k && B_col + 3 < m);

        // tileB[By][Bx] = valid_B0 ? B[B_row * m + B_col] : 0.0f;
        // tileB[By][Bx + 1] = valid_B1 ? B[B_row * m + B_col + 1] : 0.0f;
        // tileB[By][Bx + 2] = valid_B2 ? B[B_row * m + B_col + 2] : 0.0f;
        // tileB[By][Bx + 3] = valid_B3 ? B[B_row * m + B_col + 3] : 0.0f;
        
        __syncthreads();
        
        for (int i = 0; i < KS; i++) {
            C_reg[0][0].x += tileA[threadIdx.y * 8][i] * tileB[i][threadIdx.x * 8];
            C_reg[0][0].y += tileA[threadIdx.y * 8][i] * tileB[i][threadIdx.x * 8 + 1];
            C_reg[0][0].z += tileA[threadIdx.y * 8][i] * tileB[i][threadIdx.x * 8 + 2];
            C_reg[0][0].w += tileA[threadIdx.y * 8][i] * tileB[i][threadIdx.x * 8 + 3];

            C_reg[0][1].x += tileA[threadIdx.y * 8][i] * tileB[i][threadIdx.x * 8 + 4];
            C_reg[0][1].y += tileA[threadIdx.y * 8][i] * tileB[i][threadIdx.x * 8 + 5];
            C_reg[0][1].z += tileA[threadIdx.y * 8][i] * tileB[i][threadIdx.x * 8 + 6];
            C_reg[0][1].w += tileA[threadIdx.y * 8][i] * tileB[i][threadIdx.x * 8 + 7];

            C_reg[1][0].x += tileA[threadIdx.y * 8 + 1][i] * tileB[i][threadIdx.x * 8];
            C_reg[1][0].y += tileA[threadIdx.y * 8 + 1][i] * tileB[i][threadIdx.x * 8 + 1];
            C_reg[1][0].z += tileA[threadIdx.y * 8 + 1][i] * tileB[i][threadIdx.x * 8 + 2];
            C_reg[1][0].w += tileA[threadIdx.y * 8 + 1][i] * tileB[i][threadIdx.x * 8 + 3];

            C_reg[1][1].x += tileA[threadIdx.y * 8 + 1][i] * tileB[i][threadIdx.x * 8 + 4];
            C_reg[1][1].y += tileA[threadIdx.y * 8 + 1][i] * tileB[i][threadIdx.x * 8 + 5];
            C_reg[1][1].z += tileA[threadIdx.y * 8 + 1][i] * tileB[i][threadIdx.x * 8 + 6];
            C_reg[1][1].w += tileA[threadIdx.y * 8 + 1][i] * tileB[i][threadIdx.x * 8 + 7];

            C_reg[2][0].x += tileA[threadIdx.y * 8 + 2][i] * tileB[i][threadIdx.x * 8];
            C_reg[2][0].y += tileA[threadIdx.y * 8 + 2][i] * tileB[i][threadIdx.x * 8 + 1];
            C_reg[2][0].z += tileA[threadIdx.y * 8 + 2][i] * tileB[i][threadIdx.x * 8 + 2];
            C_reg[2][0].w += tileA[threadIdx.y * 8 + 2][i] * tileB[i][threadIdx.x * 8 + 3];

            C_reg[2][1].x += tileA[threadIdx.y * 8 + 2][i] * tileB[i][threadIdx.x * 8 + 4];
            C_reg[2][1].y += tileA[threadIdx.y * 8 + 2][i] * tileB[i][threadIdx.x * 8 + 5];
            C_reg[2][1].z += tileA[threadIdx.y * 8 + 2][i] * tileB[i][threadIdx.x * 8 + 6];
            C_reg[2][1].w += tileA[threadIdx.y * 8 + 2][i] * tileB[i][threadIdx.x * 8 + 7];

            C_reg[3][0].x += tileA[threadIdx.y * 8 + 3][i] * tileB[i][threadIdx.x * 8];
            C_reg[3][0].y += tileA[threadIdx.y * 8 + 3][i] * tileB[i][threadIdx.x * 8 + 1];
            C_reg[3][0].z += tileA[threadIdx.y * 8 + 3][i] * tileB[i][threadIdx.x * 8 + 2];
            C_reg[3][0].w += tileA[threadIdx.y * 8 + 3][i] * tileB[i][threadIdx.x * 8 + 3];

            C_reg[3][1].x += tileA[threadIdx.y * 8 + 3][i] * tileB[i][threadIdx.x * 8 + 4];
            C_reg[3][1].y += tileA[threadIdx.y * 8 + 3][i] * tileB[i][threadIdx.x * 8 + 5];
            C_reg[3][1].z += tileA[threadIdx.y * 8 + 3][i] * tileB[i][threadIdx.x * 8 + 6];
            C_reg[3][1].w += tileA[threadIdx.y * 8 + 3][i] * tileB[i][threadIdx.x * 8 + 7];

            C_reg[4][0].x += tileA[threadIdx.y * 8 + 4][i] * tileB[i][threadIdx.x * 8];
            C_reg[4][0].y += tileA[threadIdx.y * 8 + 4][i] * tileB[i][threadIdx.x * 8 + 1];
            C_reg[4][0].z += tileA[threadIdx.y * 8 + 4][i] * tileB[i][threadIdx.x * 8 + 2];
            C_reg[4][0].w += tileA[threadIdx.y * 8 + 4][i] * tileB[i][threadIdx.x * 8 + 3];

            C_reg[4][1].x += tileA[threadIdx.y * 8 + 4][i] * tileB[i][threadIdx.x * 8 + 4];
            C_reg[4][1].y += tileA[threadIdx.y * 8 + 4][i] * tileB[i][threadIdx.x * 8 + 5];
            C_reg[4][1].z += tileA[threadIdx.y * 8 + 4][i] * tileB[i][threadIdx.x * 8 + 6];
            C_reg[4][1].w += tileA[threadIdx.y * 8 + 4][i] * tileB[i][threadIdx.x * 8 + 7];

            C_reg[5][0].x += tileA[threadIdx.y * 8 + 5][i] * tileB[i][threadIdx.x * 8];
            C_reg[5][0].y += tileA[threadIdx.y * 8 + 5][i] * tileB[i][threadIdx.x * 8 + 1];
            C_reg[5][0].z += tileA[threadIdx.y * 8 + 5][i] * tileB[i][threadIdx.x * 8 + 2];
            C_reg[5][0].w += tileA[threadIdx.y * 8 + 5][i] * tileB[i][threadIdx.x * 8 + 3];

            C_reg[5][1].x += tileA[threadIdx.y * 8 + 5][i] * tileB[i][threadIdx.x * 8 + 4];
            C_reg[5][1].y += tileA[threadIdx.y * 8 + 5][i] * tileB[i][threadIdx.x * 8 + 5];
            C_reg[5][1].z += tileA[threadIdx.y * 8 + 5][i] * tileB[i][threadIdx.x * 8 + 6];
            C_reg[5][1].w += tileA[threadIdx.y * 8 + 5][i] * tileB[i][threadIdx.x * 8 + 7];
            
            C_reg[6][0].x += tileA[threadIdx.y * 8 + 6][i] * tileB[i][threadIdx.x * 8];
            C_reg[6][0].y += tileA[threadIdx.y * 8 + 6][i] * tileB[i][threadIdx.x * 8 + 1];
            C_reg[6][0].z += tileA[threadIdx.y * 8 + 6][i] * tileB[i][threadIdx.x * 8 + 2];
            C_reg[6][0].w += tileA[threadIdx.y * 8 + 6][i] * tileB[i][threadIdx.x * 8 + 3];

            C_reg[6][1].x += tileA[threadIdx.y * 8 + 6][i] * tileB[i][threadIdx.x * 8 + 4];
            C_reg[6][1].y += tileA[threadIdx.y * 8 + 6][i] * tileB[i][threadIdx.x * 8 + 5];
            C_reg[6][1].z += tileA[threadIdx.y * 8 + 6][i] * tileB[i][threadIdx.x * 8 + 6];
            C_reg[6][1].w += tileA[threadIdx.y * 8 + 6][i] * tileB[i][threadIdx.x * 8 + 7];

            C_reg[7][0].x += tileA[threadIdx.y * 8 + 7][i] * tileB[i][threadIdx.x * 8];
            C_reg[7][0].y += tileA[threadIdx.y * 8 + 7][i] * tileB[i][threadIdx.x * 8 + 1];
            C_reg[7][0].z += tileA[threadIdx.y * 8 + 7][i] * tileB[i][threadIdx.x * 8 + 2];
            C_reg[7][0].w += tileA[threadIdx.y * 8 + 7][i] * tileB[i][threadIdx.x * 8 + 3];

            C_reg[7][1].x += tileA[threadIdx.y * 8 + 7][i] * tileB[i][threadIdx.x * 8 + 4];
            C_reg[7][1].y += tileA[threadIdx.y * 8 + 7][i] * tileB[i][threadIdx.x * 8 + 5];
            C_reg[7][1].z += tileA[threadIdx.y * 8 + 7][i] * tileB[i][threadIdx.x * 8 + 6];
            C_reg[7][1].w += tileA[threadIdx.y * 8 + 7][i] * tileB[i][threadIdx.x * 8 + 7];
        }
        
        __syncthreads();
    }
    
    int C_row = blockIdx.y * TILE_SIZE + threadIdx.y * 8;
    int C_col = blockIdx.x * TILE_SIZE + threadIdx.x * 8;

    if (C_row < n && C_col < m) {
        *((float4*)(&C[C_row * m + C_col])) = C_reg[0][0];
        *((float4*)(&C[(C_row + 1) * m + C_col])) = C_reg[1][0];
        *((float4*)(&C[(C_row + 2) * m + C_col])) = C_reg[2][0];
        *((float4*)(&C[(C_row + 3) * m + C_col])) = C_reg[3][0];
        *((float4*)(&C[(C_row + 4) * m + C_col])) = C_reg[4][0];
        *((float4*)(&C[(C_row + 5) * m + C_col])) = C_reg[5][0];
        *((float4*)(&C[(C_row + 6) * m + C_col])) = C_reg[6][0];
        *((float4*)(&C[(C_row + 7) * m + C_col])) = C_reg[7][0];
    }

    if (C_row < n && C_col + 4 < m) {
        *((float4*)(&C[C_row * m + C_col + 4])) = C_reg[0][1];
        *((float4*)(&C[(C_row + 1) * m + C_col + 4])) = C_reg[1][1];
        *((float4*)(&C[(C_row + 2) * m + C_col + 4])) = C_reg[2][1];
        *((float4*)(&C[(C_row + 3) * m + C_col + 4])) = C_reg[3][1];
        *((float4*)(&C[(C_row + 4) * m + C_col + 4])) = C_reg[4][1];
        *((float4*)(&C[(C_row + 5) * m + C_col + 4])) = C_reg[5][1];
        *((float4*)(&C[(C_row + 6) * m + C_col + 4])) = C_reg[6][1];
        *((float4*)(&C[(C_row + 7) * m + C_col + 4])) = C_reg[7][1];
    }
}