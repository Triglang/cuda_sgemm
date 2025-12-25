#include <cstdio>
#include <cstdint>
#include <ctime>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cublas_v2.h>
#include <cuda_runtime.h>

using namespace std;

void *my_malloc(uint64_t size)
{
    void *ptr;
    const size_t alignment = 64;
    if (posix_memalign(&ptr, alignment, size) != 0)
        return nullptr;
    return ptr;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_file> <answer_file>\n", argv[0]);
        return 1;
    }

    // 读取输入文件
    FILE* in_file = fopen(argv[1], "rb");
    if (!in_file) {
        fprintf(stderr, "Failed to open input file\n");
        return 1;
    }

    int n, m, k;
    if (fread(&n, sizeof(int), 1, in_file) != 1 ||
        fread(&m, sizeof(int), 1, in_file) != 1 ||
        fread(&k, sizeof(int), 1, in_file) != 1) {
        fprintf(stderr, "Failed to read matrix dimensions\n");
        fclose(in_file);
        return 1;
    }

    // 分配内存并读取矩阵数据
    float* A = static_cast<float*>(my_malloc(n * k * sizeof(float)));
    float* B = static_cast<float*>(my_malloc(k * m * sizeof(float)));
    float* C = static_cast<float*>(my_malloc(n * m * sizeof(float)));

    if (fread(A, sizeof(float), n*k, in_file) != (size_t)(n*k) ||
        fread(B, sizeof(float), k*m, in_file) != (size_t)(k*m)) {
        fprintf(stderr, "Failed to read matrix data\n");
        fclose(in_file);
        return 1;
    }
    fclose(in_file);

    // 输出当前时间、矩阵维度、进程数
    time_t now = time(nullptr);
    tm* tm_info = localtime(&now);
    char time_str[100];
    strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", tm_info);
    printf("当前时间: %s\n矩阵维度: n=%d m=%d k=%d\n", time_str, n, m, k);

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, n * k * sizeof(float));
    cudaMalloc((void**)&d_B, k * m * sizeof(float));
    cudaMalloc((void **)&d_C, n * m * sizeof(float));

    cudaMemcpy(d_A, A, n * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, k * m * sizeof(float), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    constexpr int warm_up_count = 10;
    for (int i = 0; i < warm_up_count; i++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_B, m, d_A, k, &beta, d_C, m);
    }

    constexpr int iter_count = 10;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    for (int i = 0; i < iter_count; i++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_B, m, d_A, k, &beta, d_C, m);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    float elapsed_seconds = elapsed_ms / 1000.0f / iter_count;
    
    printf("计算耗时: %.6f秒 (%.3f毫秒)，迭代次数: %d\n", elapsed_seconds, elapsed_ms, iter_count);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(C, d_C, n * m * sizeof(float), cudaMemcpyDeviceToHost);

    // 读取标准答案
    FILE* ans_file = fopen(argv[2], "rb");
    if (!ans_file) {
        fprintf(stderr, "Failed to open answer file\n");
        return 1;
    }
    
    float* C_std = static_cast<float*>(my_malloc(n * m * sizeof(float)));
    if (fread(C_std, sizeof(float), n*m, ans_file) != (size_t)(n*m)) {
        fprintf(stderr, "Failed to read standard answer\n");
        fclose(ans_file);
        return 1;
    }
    fclose(ans_file);

    // 结果校验
    bool all_correct = true;
    float max_rel_error = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            int idx = i * m + j;
            float val = C[idx];
            float std_val = C_std[idx];

            // 检查NaN
            if (isnan(val)) {
                printf("错误: C[%d][%d] 结果为NaN\n", i, j);
                all_correct = false;
                continue;
            }

            // 计算误差
            float abs_err = fabsf (val - std_val);
            float rel_err = (std_val == 0) ? abs_err : min(abs_err / fabsf (std_val), abs_err);

            // 更新最大校验误差
            if (rel_err > max_rel_error) {
                max_rel_error = rel_err;
            }

            // 误差校验
            if (rel_err > 1e-6) {
                // printf("误差过大: C[%d][%d] 计算值=%.3e 标准值=%.3e (绝对误差=%.3e 校验误差=%.3e)\n",
                //         i, j, val, std_val, abs_err, rel_err);
                all_correct = false;
            }
        }
    }

    long long total_ops = 2LL * n * m * k;
    double flops = (double)total_ops / elapsed_seconds;
    double gflops = flops / 1e9;  // 转换为 GFLOPS (Giga FLOPS)

    // 输出最终结果
    if (all_correct) {
        printf("结果正确，最大校验误差: %.3e\n", max_rel_error);
        printf("GFLOPS: %f\n", gflops);
    } else {
        printf("存在错误，最大校验误差: %.3e\n", max_rel_error);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}