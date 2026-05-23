#pragma once

#include <cuda_runtime.h>

/*
	A matrix designed to have (most) operations run on the GPU via CUDA kernels.

	This matrix is row major, and (for now) only supports floats.
*/
struct SCMatrix
{
    SCMatrix(size_t h, size_t w)
        : height(h),
          width(w),
          num_elements(h * w)
    {
        cudaMalloc(&data_, num_elements * sizeof(float));
    }

    ~SCMatrix()
    {
        if (data_)
            cudaFree(data_);
    }

    SCMatrix(const SCMatrix&) = delete;
    SCMatrix& operator=(const SCMatrix&) = delete;

    SCMatrix(SCMatrix&& other) noexcept
        : height(other.height),
          width(other.width),
          num_elements(other.num_elements),
          data_(other.data_)
    {
        other.data_ = nullptr;
    }

    float* data_ = nullptr;

    const size_t height, width, num_elements;
};