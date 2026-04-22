#include "sgemm.h"

void launch_sgemm(SgemmKernel k, const SgemmParams& p)
{
    switch (k)
    {
        case SgemmKernel::CuBLAS:
            sgemm_cublas(p);
            break;

        case SgemmKernel::Naive:
            sgemm_naive(p);
            break;

        case SgemmKernel::GlobalCoalescing:
            sgemm_global_coalescing(p);
            break;
    }
}