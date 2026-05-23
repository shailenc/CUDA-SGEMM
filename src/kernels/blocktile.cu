#include "sgemm.h"

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

constexpr int32_t BLOCK_WIDTH = 32;
constexpr int32_t BLOCK_HEIGHT = 32;

constexpr int32_t TM = 2;
constexpr int32_t TN = 2;

__global__ void sgemm_blocktile
(
	int M, int N, int K, float alpha,
	const float* A, const float* B, float beta, float* C
)
{
    // '(0,0)' within a subtile.
    // X is COLUMN NUMBER
    // Y IS ROW NUMBER
    const uint32_t row = (blockIdx.y * blockDim.y + threadIdx.y) * TM;
    const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) * TN;

    __shared__ float As[BLOCK_HEIGHT][BLOCK_WIDTH];
    __shared__ float Bs[BLOCK_HEIGHT][BLOCK_WIDTH];

    float threadResults[TM * TN];

    // init thread result registers
    #pragma unroll
    for (int i = 0; i < TM*TN; ++i) {
        threadResults[i] = 0.0f;
    }

    // load into smem
    for (int tileK = 0; tileK < K; tileK += BLOCK_WIDTH) {
        
        // Load A and B into SMEM

        for (int i = 0; i < TM; ++i) {
            for (int j = 0; j < TN; ++j) {
                int r = row + i;
                int c = tileK + threadIdx.x * TN + j;
                
                As[threadIdx.y * TM + i][threadIdx.x * TN + j] =
                    (r < M && c < K) ? A[r * K + c] : 0.0f;
            }
        }

        for (int i = 0; i < TN; ++i) {
            for (int j = 0; j < TN; ++j) {
                int r = tileK + threadIdx.y * TM + i;
                int c = col + j;
                
                Bs[threadIdx.y * TM + i][threadIdx.x * TN + j] =
                    (r < K && c < N) ? B[r*N + c] : 0.0f;
            }
        }

        __syncthreads();

        // Calc partial dot products
        for (int k = 0; k < BLOCK_WIDTH; ++k) {
            float a[TM];
            float b[TN];

            for (int i = 0; i < TM; ++i)
                a[i] = As[threadIdx.y * TM + i][k];

            for (int j = 0; j < TN; ++j)
                b[j] = Bs[k][threadIdx.x * TN + j];

            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j)
                    threadResults[i * TN + j] += a[i] * b[j];
        }

        __syncthreads();
    }

    for (int i = 0; i < TM; ++i) {
        int r = row + i;
        if (r >= M) continue;

        for (int j = 0; j < TN; j++) {
            int c = col + j;
            if (c >= N) continue;

            int idx = r * N + c;

            C[idx] = alpha * threadResults[i * TN + j] + beta * C[idx];
        }
    }
}

void sgemm_blocktile(const SgemmParams &p)
{
    dim3 block(BLOCK_WIDTH / TN, BLOCK_HEIGHT / TM);
    dim3 grid(CEIL_DIV(p.N, BLOCK_WIDTH), CEIL_DIV(p.M, BLOCK_HEIGHT));
    sgemm_blocktile<<<grid, block>>>(p.M, p.N, p.K, p.alpha, p.A, p.B, p.beta, p.C);
}