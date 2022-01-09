#pragma once
#include <cutensor.h>
#include <cublas_v2.h>
#include <cudnn.h>


namespace karuiflow {
	class CudaContextManager {
	public:
		static cutensorHandle_t* getCuTensorHandle();
		static cublasHandle_t* getCublasHandle();
		static cudnnHandle_t* getCudnnHandle();
	private:
		static cutensorHandle_t s_CuTensorHandle;
		static bool s_CuTensorHandleInitialized;
		static cublasHandle_t s_CublasHandle;
		static bool s_CublasHandleInitialized;
		static cudnnHandle_t s_CudnnHandle;
		static bool s_CudnnHandleInitialized;
	};
}
