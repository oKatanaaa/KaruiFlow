#pragma once
#include <cuda_runtime.h>
#include <cutensor.h>
#include <vector>


namespace karuiflow {

	void setDebugLogLevel();
	void setInfoLogLevel();
	void setErrLogLevel();
	void setWarnLogLevel();

	std::string shapeToString(std::vector<int>& shape);
}

 void CudaCheck(cudaError_t error, const char* file, int line) {
	if (error != cudaSuccess)
	{
		fprintf(stderr, "Error: %s:%d, ", file, line);
		fprintf(stderr, "code: %d, reason: %s\n", error,
			cudaGetErrorString(error));
		throw std::runtime_error("Error in CUDA: " + std::string(cudaGetErrorString(error)));
	}
}

#define CUDA_CHECK( err ) (CudaCheck( err, __FILE__, __LINE__ ))


 void CuTensorCheck(cutensorStatus_t error, const char* file, int line) {
	if (error != CUTENSOR_STATUS_SUCCESS)
	{
		fprintf(stderr, "Error: %s:%d, ", file, line);
		fprintf(stderr, "code: %d, reason: %s\n", error,
			cutensorGetErrorString(error));
		throw std::runtime_error("Error in CUDA: " + std::string(cutensorGetErrorString(error)));
	}
}

#define CUTENSOR_CHECK( err ) (CuTensorCheck( err, __FILE__, __LINE__ ))
