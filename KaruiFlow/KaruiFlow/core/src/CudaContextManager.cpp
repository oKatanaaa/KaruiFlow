#include "../headers/LoggingUtils.h"
#include <spdlog/spdlog.h>
#include "../headers/CudaContextManager.h"


namespace karuiflow {
	cutensorHandle_t* CudaContextManager::getCuTensorHandle() {
		if (!s_CuTensorHandleInitialized) {
			spdlog::info("Initializzing cuTensor handle...");
			CUTENSOR_CHECK(cutensorInit(&s_CuTensorHandle));
			spdlog::info("Initialized cuTensor handle.");
			s_CuTensorHandleInitialized = true;
		}
		return &s_CuTensorHandle;
	}

	cublasHandle_t* CudaContextManager::getCublasHandle() {
		if (!s_CublasHandleInitialized) {
			spdlog::info("Initializzing Cublas handle...");
			CUBLAS_CHECK(cublasCreate(&s_CublasHandle));
			spdlog::info("Initialized Cublas handle.");
			s_CublasHandleInitialized = true;
		}
		return &s_CublasHandle;
	}

	cudnnHandle_t* CudaContextManager::getCudnnHandle() {
		if (!s_CudnnHandleInitialized) {
			spdlog::info("Initializzing Cudnn handle...");
			CUDNN_CHECK(cudnnCreate(&s_CudnnHandle));
			spdlog::info("Initialized Cudnn handle.");
			s_CudnnHandleInitialized = true;
		}
		return &s_CudnnHandle;
	}

	cutensorHandle_t CudaContextManager::s_CuTensorHandle;
	bool CudaContextManager::s_CuTensorHandleInitialized = false;

	cublasHandle_t CudaContextManager::s_CublasHandle;
	bool CudaContextManager::s_CublasHandleInitialized = false;

	cudnnHandle_t CudaContextManager::s_CudnnHandle;
	bool CudaContextManager::s_CudnnHandleInitialized = false;
}
