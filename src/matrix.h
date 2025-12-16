#pragma once

#include <type_traits>
#include <cuda_runtime.h>

/*
	A matrix designed to have (most) operations run on the GPU via CUDA kernels.

	This matrix is row major, and (for now) only supports floats.
*/
class SCMatrix
{
public:

	const size_t height;
	const size_t width;
	const size_t num_elements;

	SCMatrix(size_t input_height, size_t input_width);
	~SCMatrix();

	float& operator()(const size_t row, const size_t col);
	float operator()(const size_t row, const size_t col) const;

	SCMatrix operator*(const SCMatrix& rhs) const;

private:

	float* data_ = nullptr;
};