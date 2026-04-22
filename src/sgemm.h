#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>

enum class SgemmKernel
{
    CuBLAS,
    Naive,
    GlobalCoalescing
};

struct SgemmParams
{
    int M, N, K;
    float alpha, beta;
    const float* A;
    const float* B;
    float* C;
    cublasHandle_t handle;
};

void launch_sgemm(SgemmKernel k, const SgemmParams& p);

void sgemm_cublas(const SgemmParams& p);
void sgemm_naive(const SgemmParams& p);
void sgemm_global_coalescing(const SgemmParams& p);
