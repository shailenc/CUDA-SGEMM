#include "sgemm.h"

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

constexpr int32_t TILESIZE = 32;

__global__ void sgemm_blocktile
(
	int M, int N, int K, float alpha,
	const float* A, const float* B, float beta, float* C
)
{
    // X is COLUMN NUMBER
    // Y IS ROW NUMBER
    const uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float As[TILESIZE][TILESIZE];
    __shared__ float Bs[TILESIZE][TILESIZE];
    
    if (row < M && col < N) {

        float tmp = 0.0;

        for (int tile = 0; tile < K; tile += TILESIZE) {

            // Load data into SMEM for calcs
            if (row < M && (tile + threadIdx.x) < K)
                As[threadIdx.y][threadIdx.x] =
                    A[row * K + (tile + threadIdx.x)];

            if ((tile + threadIdx.y) < K && col < N)
                Bs[threadIdx.y][threadIdx.x] =
                    B[(tile + threadIdx.y) * N + col];

            __syncthreads();

            // Add partial sums to tmp
            #pragma unroll
            for (int k = 0; k < TILESIZE; ++k)
                tmp += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        C[row * N + col] = alpha * tmp + beta * C[row * N + col];
    
    }
}

void sgemm_blocktile(const SgemmParams &p)
{
    dim3 block(TILESIZE, TILESIZE);
    dim3 grid(CEIL_DIV(p.M, TILESIZE), CEIL_DIV(p.N, TILESIZE));
    sgemm_blocktile<<<grid, block>>>(p.M, p.N, p.K, p.alpha, p.A, p.B, p.beta, p.C);
}