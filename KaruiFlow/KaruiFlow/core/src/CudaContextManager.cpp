#include <cutensor.h>
#include "../headers/LoggingUtils.h"
#include <spdlog/spdlog.h>
#include "../headers/CudaContextManager.h"


namespace karuiflow {
	cutensorHandle_t* CudaContextManager::getCuTensorHandle() {
		if (!s_CuTensorHandleInitialized) {
			CUTENSOR_CHECK(cutensorInit(&s_CuTensorHandle));
			spdlog::info("Initialized cuTensor handle.");
			s_CuTensorHandleInitialized = true;
		}
		return &s_CuTensorHandle;
	}

	cutensorHandle_t CudaContextManager::s_CuTensorHandle;
	bool CudaContextManager::s_CuTensorHandleInitialized = false;
}
