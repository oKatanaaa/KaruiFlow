#pragma once
#include <cutensor.h>


namespace karuiflow {
	class CudaContextManager {
	public:
		static cutensorHandle_t* getCuTensorHandle();

	private:
		static cutensorHandle_t s_CuTensorHandle;
		static bool s_CuTensorHandleInitialized;
	};
}
