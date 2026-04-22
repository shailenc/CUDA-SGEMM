#include "sgemm.h"

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// This is nearly identical to the naive kernel, but with
//  x and y swapped.
__global__ void sgemm_global_coalescing
(
	int M, int N, int K, float alpha,
	const float* A, const float* B, float beta, float* C
)
{
    const uint32_t x = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t y = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
        tmp += A[x * K + i] * B[i * N + y];
    }
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

void sgemm_global_coalescing(const SgemmParams &p)
{
    dim3 block(32, 32);
    dim3 grid(CEIL_DIV(p.M, 32), CEIL_DIV(p.N, 32));
    sgemm_global_coalescing<<<grid, block>>>(p.M, p.N, p.K, p.alpha, p.A, p.B, p.beta, p.C);
}