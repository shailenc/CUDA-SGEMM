#include "matrix.h"
#include "sgemm.h"

#include <iostream>
#include <vector>

static float benchmark_sgemm(SgemmKernel k,
                             const SgemmParams& p,
                             int warmup = 10,
                             int runs = 100)
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

static bool checkSgemmCorrectness(SgemmKernel k)
{
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    SCMatrix A(M, K);
    SCMatrix B(K, N);
    SCMatrix C(M, N);

    // host buffers
    std::vector<float> hA(M * K, 1.0f);
    std::vector<float> hB(K * N, 1.0f);
    std::vector<float> hC(M * N, 0.0f);

    // copy to device
    cudaMemcpy(A.data_, hA.data(), sizeof(float) * hA.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(B.data_, hB.data(), sizeof(float) * hB.size(), cudaMemcpyHostToDevice);
    cudaMemset(C.data_, 0, sizeof(float) * hC.size());

    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
        return false;

    SgemmParams params{
        M, N, K,
        1.0f, 0.0f,
        A.data_,
        B.data_,
        C.data_,
        handle
    };

    launch_sgemm(k, params);
    cudaDeviceSynchronize();

    // copy result back
    cudaMemcpy(hC.data(), C.data_, sizeof(float) * hC.size(), cudaMemcpyDeviceToHost);

    const float expected = static_cast<float>(K);
    const float eps = 1e-3f;
    for (size_t i = 0; i < hC.size(); ++i)
    {
        if (std::fabs(hC[i] - expected) > eps)
        {
            cublasDestroy(handle);
            return false;
        }
    }

    cublasDestroy(handle);
    return true;
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
        SgemmKernel::Naive,
        SgemmKernel::GlobalCoalescing,
        SgemmKernel::SMEM,
        SgemmKernel::Blocktile
    };

    for (SgemmKernel k: kernels)
    {
        float ms = benchmark_sgemm(k, params);

        double flops = 2.0 * M * N * K;
        double gflops = (flops / (ms * 1e-3)) / 1e9;

        std::cout << "Kernel " << static_cast<int>(k)+1
                  << "\t| " << ms << " ms"
                  << "\t| " << gflops << " GFLOP/s\t";

        if (!checkSgemmCorrectness(k)) {
            std::cout << "incorrect!";
        }

        std::cout << "\n";
    }

    cublasDestroy(handle);

    return 0;
}