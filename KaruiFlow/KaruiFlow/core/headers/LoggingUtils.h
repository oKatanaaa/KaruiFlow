#pragma once
#include <cuda_runtime.h>
#include <cutensor.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <vector>
#include <string>
#include <stdexcept>


namespace karuiflow {

	void setDebugLogLevel();
	void setInfoLogLevel();
	void setErrLogLevel();
	void setWarnLogLevel();

	std::string shapeToString(std::vector<int>& shape);

	void CudaCheck(cudaError_t error, const char* file, int line);
	void CuTensorCheck(cutensorStatus_t error, const char* file, int line);
	void CublasCheck(cublasStatus_t error, const char* file, int line);
	void CudnnCheck(cudnnStatus_t error, const char* file, int line);
}

#define CUDA_CHECK( err ) (CudaCheck( err, __FILE__, __LINE__ ))
#define CUTENSOR_CHECK( err ) (CuTensorCheck( err, __FILE__, __LINE__ ))
#define CUBLAS_CHECK( err ) (CublasCheck( err, __FILE__, __LINE__ ))
#define CUDNN_CHECK( err ) (CudnnCheck( err, __FILE__, __LINE__ ))
