#include "../headers/LoggingUtils.h"
#include <spdlog/spdlog.h>
#include "../headers/CudaContextManager.h"


namespace karuiflow {
	cutensorHandle_t* CudaContextManager::getCuTensorHandle() {
		spdlog::info("Initializzing cuTensor handle...");
		if (!s_CuTensorHandleInitialized) {
			CUTENSOR_CHECK(cutensorInit(&s_CuTensorHandle));
			spdlog::info("Initialized cuTensor handle.");
			s_CuTensorHandleInitialized = true;
		}
		return &s_CuTensorHandle;
	}

	cublasHandle_t* CudaContextManager::getCublasHandle() {
		spdlog::info("Initializzing Cublas handle...");
		if (!s_CublasHandleInitialized) {
			CUBLAS_CHECK(cublasCreate(&s_CublasHandle));
			spdlog::info("Initialized Cublas handle.");
		}
		return &s_CublasHandle;
	}

	cutensorHandle_t CudaContextManager::s_CuTensorHandle;
	bool CudaContextManager::s_CuTensorHandleInitialized = false;

	cublasHandle_t CudaContextManager::s_CublasHandle;
	bool CudaContextManager::s_CublasHandleInitialized = false;
}
