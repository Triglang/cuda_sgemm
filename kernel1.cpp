#include <cuda_runtime.h>

__global__ void gemm_kernel1(float *A, float *B, float *C, int n, int m, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < m)
    {
        float sum = 0.0;
        for (int i = 0; i < k; ++i)
        {
            sum += A[row * k + i] * B[i * m + col];
        }
        C[row * m + col] = sum;
    }
}