#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>


template<typename T>
__global__ void cudaAdd(T* x, T* y, T* out)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	out[tid] = x[tid] + y[tid];
}

const int THREADS_PER_BLOCK = 128;


template<typename T>
void add(void* _x, void* _y, void* _out, size_t n) {
	T* x = (T*)_x;
	T* y = (T*)_y;
	T* out = (T*)_out;
	int n_blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	cudaAdd<T> << <n_blocks, THREADS_PER_BLOCK >> > (x, y, out);
}
