#include "sgemm.h"
#include <cublas_v2.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

void sgemm_cublas(const SgemmParams &p)
{
    const float alpha = p.alpha;
    const float beta = p.beta;

    // Matrices are row-major. To compute C = alpha * A * B + beta * C
    // with cuBLAS (which expects column-major), call sgemm with A and B
    // swapped and dimensions (m,n,k) = (N, M, K).
    cublasSgemm(p.handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                p.N, p.M, p.K,
                &alpha,
                p.B, p.N,
                p.A, p.K,
                &beta,
                p.C, p.N);
}