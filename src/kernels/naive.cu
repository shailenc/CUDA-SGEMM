#include "sgemm.h"

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

__global__ void sgemm_naive
(
	int M, int N, int K, float alpha,
	const float* A, const float* B, float beta, float* C
)
{
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= M || y >= N)
		return;

	float tmp = 0.0f;

	for (int i = 0; i < K; ++i)
		tmp += A[x * K + i] * B[i * N + y];

	C[x * N + y] = alpha * tmp + beta * C[x * N + y];
}

void sgemm_naive(const SgemmParams &p)
{
    dim3 block(32, 32);
    dim3 grid(CEIL_DIV(p.M, 32), CEIL_DIV(p.N, 32));
    sgemm_naive<<<grid, block>>>(p.M, p.N, p.K, p.alpha, p.A, p.B, p.beta, p.C);
}