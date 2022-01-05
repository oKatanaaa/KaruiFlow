#pragma once
#include <cutensor.h>
#include <cublas_v2.h>
#include <spdlog/spdlog.h>
#include "../headers/LoggingUtils.h"


namespace karuiflow {
	class CudaContextManager {
	public:
		static cutensorHandle_t*  getCuTensorHandle() {
			if (!s_CuTensorHandleInitialized) {
				CUTENSOR_CHECK(cutensorInit(&s_CuTensorHandle));
				spdlog::info("Initialized cuTensor handle.");
			}
			return &s_CuTensorHandle;
		}

		static cublasHandle_t* getCublasHandle() {
			if (!s_CublasHandleInitialized) {
				CUBLAS_CHECK(cublasCreate(&s_CublasHandle));
				spdlog::info("Initialized Cublas handle.");
			}
			return &s_CublasHandle;
		}
	private:
		static cutensorHandle_t s_CuTensorHandle;
		static bool s_CuTensorHandleInitialized;
		static cublasHandle_t s_CublasHandle;
		static bool s_CublasHandleInitialized;
	};

	bool CudaContextManager::s_CuTensorHandleInitialized = false;
}
