#include "../headers/memory/DeviceCUDA.h"
#include "../headers/memory/Exceptions.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "DeviceCudaAdder.h"

static void CudaCheck(cudaError_t error, const char* file, int line) {
	if (error != cudaSuccess)
	{
		fprintf(stderr, "Error: %s:%d, ", file, line);
		fprintf(stderr, "code: %d, reason: %s\n", error,
			cudaGetErrorString(error));
		throw std::runtime_error("Error in CUDA: " + std::string(cudaGetErrorString(error)));
	}
}

#define CUDA_CHECK( err ) (CudaCheck( err, __FILE__, __LINE__ ))




namespace karuiflow {
	

	void DeviceCUDA::allocateMemory(void** ptr, size_t bytes) {
		CUDA_CHECK(cudaMalloc(ptr, bytes));
	}

	void DeviceCUDA::deallocateMemory(void* ptr) {
		CUDA_CHECK(cudaFree(ptr));
	}

	void DeviceCUDA::copyDeviceToCpu(void* src, void* dst, size_t bytes) {
		CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
	}

	void DeviceCUDA::copyCpuToDevice(void* src, void* dst, size_t bytes) {
		CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
	}

	void DeviceCUDA::copyDeviceToDevice(void* src, void* dst, size_t bytes) {
		CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
	}

	void DeviceCUDA::setZeros(void* src, size_t bytes) {
		CUDA_CHECK(cudaMemset(src, 0, bytes));
	}

	std::string DeviceCUDA::getDeviceName() {
		return "cuda";
	}

	int DeviceCUDA::getDeviceId() {
		return 0;
	}

	std::function<void(void*, void*, void*, size_t)> DeviceCUDA::getAdder(DType* dtype) {
		if (dtype->getName() == karuiflow::Float32().getName())
			return add<float>;
		else if (dtype->getName() == karuiflow::Int32().getName())
			return add<int>;
	}

}
