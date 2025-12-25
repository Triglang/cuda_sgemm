#ifndef KERNEL_H
#define KERNEL_H

__global__ void gemm_kernel1(float *A, float *B, float *C, int n, int m, int k);
__global__ void gemm_kernel2(float *A, float *B, float *C, int n, int m, int k);
__global__ void gemm_kernel3(float *A, float *B, float *C, int n, int m, int k);
__global__ void gemm_kernel4(float *A, float *B, float *C, int n, int m, int k);
__global__ void gemm_kernel5(float *A, float *B, float *C, int n, int m, int k);

#endif