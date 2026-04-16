#include "matrix.h"
#include "sgemm.h"

#include <iostream>
#include <vector>

static float benchmark_sgemm(SgemmKernel k,
                             const SgemmParams& p,
                             int warmup = 5,
                             int runs = 20)
{
    for (int i = 0; i < warmup; ++i)
        launch_sgemm(k, p);

    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < runs; ++i)
        launch_sgemm(k, p);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / runs;
}

int main()
{
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    SCMatrix A(M, K);
    SCMatrix B(K, N);
    SCMatrix C(M, N);

    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        return 1;
    }

    SgemmParams params {
        M, N, K,
        1.0f, 0.0f,
        A.data_,
        B.data_,
        C.data_,
        handle
    };

    std::vector<SgemmKernel> kernels = {
        SgemmKernel::CuBLAS,
        SgemmKernel::Naive
    };

    for (auto k : kernels)
    {
        float ms = benchmark_sgemm(k, params);

        double flops = 2.0 * M * N * K;
        double gflops = (flops / (ms * 1e-3)) / 1e9;

        std::cout << "Kernel " << static_cast<int>(k)+1
                  << "\t| " << ms << " ms"
                  << "\t| " << gflops << " GFLOP/s\n";
    }

    cublasDestroy(handle);

    return 0;
}