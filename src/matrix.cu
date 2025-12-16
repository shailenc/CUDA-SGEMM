#pragma once

#include "matrix.h"

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

// implementation at bottom of file.
__global__ void sgemm
(
	int M, int N, int K, float alpha,
	const float* A, const float* B, float beta, float* C
);

SCMatrix::SCMatrix(size_t input_height, size_t input_width)
	: height(input_height)
	, width(input_width)
	, num_elements(input_height*input_width)
{
	// TODO: support other types than float
	cudaMallocManaged(&data_, num_elements * sizeof(float));

	for (size_t i = 0; i < num_elements; ++i) {
		data_[i] = i;
	}
}

SCMatrix::~SCMatrix()
{
	cudaFree(data_);
}

float& SCMatrix::operator()(const size_t row, const size_t col)
{
	return data_[row * width + col];
}

float SCMatrix::operator()(const size_t row, const size_t col) const
{
	return data_[row * width + col];
}

SCMatrix SCMatrix::operator*(const SCMatrix& rhs) const
{
	const size_t out_height = this->height;
	const size_t out_width = rhs.width;

	SCMatrix result(out_height, out_width);
		
	dim3 grid_dim = dim3(CEIL_DIV(out_height, 32), CEIL_DIV(out_width, 32));
	dim3 block_dim = dim3(32, 32);

	sgemm<<<grid_dim, block_dim>>>(
		this->height, rhs.width, rhs.height, 1.0f,
		this->data_, rhs.data_, 1.0f, result.data_);

	cudaDeviceSynchronize();

	return result;
}

__global__ void sgemm
(
	int M, int N, int K, float alpha,
	const float* A, const float* B, float beta, float* C
)
{
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= M || y >= N)
		return;

	float tmp = 0.0;

	for (int i = 0; i < K; ++i)
		tmp += A[x * K + i] * B[i * N + y];

	C[x * N + y] = alpha * tmp + beta * C[x * N + y];
}