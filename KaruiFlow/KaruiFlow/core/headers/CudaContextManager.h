#pragma once
#include <cutensor.h>
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

	private:
		static cutensorHandle_t s_CuTensorHandle;
		static bool s_CuTensorHandleInitialized;
	};

	bool CudaContextManager::s_CuTensorHandleInitialized = false;
}
