#pragma once

#include <cuda_runtime.h>

/*
	A matrix designed to have (most) operations run on the GPU via CUDA kernels.

	This matrix is row major, and (for now) only supports floats.
*/
struct SCMatrix
{
	// TODO: support other types than float?
	SCMatrix(size_t input_height, size_t input_width)
	: height(input_height)
	, width(input_width)
	, num_elements(input_height*input_width)
	{
		cudaMalloc(&data_, num_elements * sizeof(float));
	}

	~SCMatrix()
	{
		cudaFree(data_);
	}


	const size_t height, width;
	const size_t num_elements;
	float* data_ = nullptr;
};