#pragma once
#include <cutensor.h>
#include <cublas_v2.h>


namespace karuiflow {
	class CudaContextManager {
	public:
		static cutensorHandle_t* getCuTensorHandle();
		static cublasHandle_t* getCublasHandle();
	private:
		static cutensorHandle_t s_CuTensorHandle;
		static bool s_CuTensorHandleInitialized;
		static cublasHandle_t s_CublasHandle;
		static bool s_CublasHandleInitialized;
	};
}
