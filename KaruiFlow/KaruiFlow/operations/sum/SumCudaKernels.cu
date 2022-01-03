#include "SumCUDAKernels.h"
#include <cutensor.h>
#include "../../core/headers/CudaContextManager.h"
#include "../../core/headers/LoggingUtils.h"


namespace karuiflow {
	void SumCudaKernel::forward(std::vector<Storage*> inputs, Storage* output) {
		void* inputData = inputs[0]->getData();
		void* outputData = output->getData();
		cutensorHandle_t* handle = CudaContextManager::getCuTensorHandle();

		// Creating a descriptor for the Tensor to reduce
		std::vector<int> tensorADim = inputs[0]->getShape();
		std::vector<long long> extentA(tensorADim.begin(), tensorADim.end());
		std::vector<int> modeA;
		for (int i = 0; i < tensorADim.size(); i++) {
			modeA.push_back(i);
		}
		cudaDataType_t cuDtypeA;
		std::string dtypeA = inputs[0]->getDtype()->getName();
		if (dtypeA == "float32")
			cuDtypeA = CUDA_R_32F;
		else if (dtypeA == "int32")
			cuDtypeA = CUDA_R_32I;
		else
			throw std::runtime_error("SumCudaKernel.forward // Expected float32 or int32, but received" + dtypeA);

		cutensorTensorDescriptor_t descA;
		CUTENSOR_CHECK(cutensorInitTensorDescriptor(
			handle,
			&descA,
			modeA.size(),
			extentA.data(),
			NULL,
			cuDtypeA,
			CUTENSOR_OP_IDENTITY
			));

		std::vector<int>tensorCDim = output->getShape();
		std::vector<int>modeC;
		

		// Creating a descriptor for the output Tensor
		for (int i = 0; i < modeA.size(); i++) {
			if (std::find(modeA.begin(), modeA.end(), i) == modeA.end())
				modeC.push_back(i);
		}


	}
}
