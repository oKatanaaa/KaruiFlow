#include "SumCUDAKernels.h"
#include <cutensor.h>
#include "../../core/headers/CudaContextManager.h"
#include "../../core/headers/LoggingUtils.h"
#include <spdlog/spdlog.h>


namespace karuiflow {
	void SumCudaKernel::forward(std::vector<Storage*> inputs, Storage* output) {
		void* d_A = inputs[0]->getData();
		void* d_C = output->getData();
		cutensorHandle_t* handle = CudaContextManager::getCuTensorHandle();

		// Creating a descriptor for the Tensor to reduce
		spdlog::debug("SumCudaKernel.forward // Creating descriptors for tensors...");
		std::vector<int> tensorADim = inputs[0]->getShape();
		std::vector<long long> extentA(tensorADim.begin(), tensorADim.end());
		std::vector<int> modeA;
		for (int i = 0; i < tensorADim.size(); i++) {
			modeA.push_back(i);
		}
		cudaDataType_t cuDtypeA;
		cutensorComputeType_t typeCompute;
		std::string dtypeA = inputs[0]->getDtype()->getName();
		if (dtypeA == "float32") {
			cuDtypeA = CUDA_R_32F;
			typeCompute = CUTENSOR_COMPUTE_32F;
		}
		else if (dtypeA == "int32") {
			cuDtypeA = CUDA_R_32I;
			typeCompute = CUTENSOR_COMPUTE_32I;
		}
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

		// Creating a descriptor for the output Tensor
		std::vector<int> tensorCDim = output->getShape();
		std::vector<int> modeC;
		std::vector<long long> extentC;
		for (int i = 0; i < modeA.size(); i++) {
			if (std::find(m_Dim.begin(), m_Dim.end(), i) == m_Dim.end()) {
				modeC.push_back(i);
				extentC.push_back(extentA[i]);
			}
		}
		cudaDataType_t cuDtypeC;
		std::string dtypeC = inputs[0]->getDtype()->getName();
		if (dtypeC == "float32")
			cuDtypeC = CUDA_R_32F;
		else if (dtypeC == "int32")
			cuDtypeC = CUDA_R_32I;
		else
			throw std::runtime_error("SumCudaKernel.forward // Expected float32 or int32, but received" + dtypeC);
		cutensorTensorDescriptor_t descC;
		CUTENSOR_CHECK(cutensorInitTensorDescriptor(
			handle,
			&descC,
			modeC.size(),
			extentC.data(),
			NULL,
			cuDtypeC,
			CUTENSOR_OP_IDENTITY
		));
		spdlog::debug("SumCudaKernel.forward // Finished creating descriptors.");

		// Querry workspace
		spdlog::debug("SumCudaKernel.forward // Quering workspace...");
		const cutensorOperator_t opReduce = CUTENSOR_OP_ADD;
		uint64_t worksize = 0;
		CUTENSOR_CHECK(cutensorReductionGetWorkspace(handle,
			d_A, &descA, modeA.data(),
			d_C, &descC, modeC.data(),
			d_C, &descC, modeC.data(),
			opReduce, typeCompute, &worksize));

		void* work = nullptr;
		
		if (worksize > 0)
			output->getDevice()->allocateMemory(&work, worksize);
		float alpha = 1.0f;
		float beta = 0.f;
		spdlog::debug("SumCudaKernel.forward // Workspace acquired.");

		// Perform compute
		spdlog::debug("SumCudaKernel.forward // Computing sum...");
		CUTENSOR_CHECK(cutensorReduction(handle,
			(const void*)&alpha, d_A, &descA, modeA.data(),
			(const void*)&beta,  d_C, &descC, modeC.data(),
								 d_C, &descC, modeC.data(),
			opReduce, typeCompute, work, worksize, 0 /* stream */));

		if (work != nullptr)
			output->getDevice()->deallocateMemory(work);

		spdlog::debug("SumCudaKernel.forward // Successfully computed sum.");
	}

	void SumCudaKernel::backward(std::vector<Storage*> inputs, std::vector<bool> requiresGrad,
		Storage* outerGradient, std::vector<Storage*> outputGradients) {
		// Example: multiplication is done via broadcasting
		// C = A * B
		// A - [dim1, dim2]
		// B - [dim1, 1]
		void* d_A = inputs[0]->getData();
		void* d_B = outerGradient->getData();
		void* d_C = outputGradients[0]->getData();
		cutensorHandle_t* handle = CudaContextManager::getCuTensorHandle();

		// Creating a descriptor for the Tensor to reduce
		spdlog::debug("SumCudaKernel.backward // Creating descriptors for tensors...");
		std::vector<int> tensorADim = inputs[0]->getShape();
		std::vector<long long> extentA(tensorADim.begin(), tensorADim.end());
		std::vector<int> modeA;
		for (int i = 0; i < tensorADim.size(); i++) {
			modeA.push_back(i);
		}
		cudaDataType_t cuDtypeA;
		cutensorComputeType_t typeCompute;
		std::string dtypeA = inputs[0]->getDtype()->getName();
		if (dtypeA == "float32") {
			cuDtypeA = CUDA_R_32F;
			typeCompute = CUTENSOR_COMPUTE_32F;
		}
		else if (dtypeA == "int32") {
			cuDtypeA = CUDA_R_32I;
			typeCompute = CUTENSOR_COMPUTE_32I;
		}
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

		// Creating a descriptor for the output Tensor
		std::vector<int> tensorCDim = output->getShape();
		std::vector<int> modeC;
		std::vector<long long> extentC;
		for (int i = 0; i < modeA.size(); i++) {
			if (std::find(m_Dim.begin(), m_Dim.end(), i) == m_Dim.end()) {
				modeC.push_back(i);
				extentC.push_back(extentA[i]);
			}
		}
		cudaDataType_t cuDtypeC;
		std::string dtypeC = inputs[0]->getDtype()->getName();
		if (dtypeC == "float32")
			cuDtypeC = CUDA_R_32F;
		else if (dtypeC == "int32")
			cuDtypeC = CUDA_R_32I;
		else
			throw std::runtime_error("SumCudaKernel.forward // Expected float32 or int32, but received" + dtypeC);
		cutensorTensorDescriptor_t descC;
		CUTENSOR_CHECK(cutensorInitTensorDescriptor(
			handle,
			&descC,
			modeC.size(),
			extentC.data(),
			NULL,
			cuDtypeC,
			CUTENSOR_OP_IDENTITY
		));
		spdlog::debug("SumCudaKernel.forward // Finished creating descriptors.");

		// Querry workspace
		spdlog::debug("SumCudaKernel.forward // Quering workspace...");
		const cutensorOperator_t opReduce = CUTENSOR_OP_ADD;
		uint64_t worksize = 0;
		CUTENSOR_CHECK(cutensorReductionGetWorkspace(handle,
			d_A, &descA, modeA.data(),
			d_C, &descC, modeC.data(),
			d_C, &descC, modeC.data(),
			opReduce, typeCompute, &worksize));

		void* work = nullptr;

		if (worksize > 0)
			output->getDevice()->allocateMemory(&work, worksize);
		float alpha = 1.0f;
		float beta = 0.f;
		spdlog::debug("SumCudaKernel.forward // Workspace acquired.");

		// Perform compute
		spdlog::debug("SumCudaKernel.forward // Computing sum...");
		CUTENSOR_CHECK(cutensorReduction(handle,
			(const void*)&alpha, d_A, &descA, modeA.data(),
			(const void*)&beta, d_C, &descC, modeC.data(),
			d_C, &descC, modeC.data(),
			opReduce, typeCompute, work, worksize, 0 /* stream */));

		if (work != nullptr)
			output->getDevice()->deallocateMemory(work);

		spdlog::debug("SumCudaKernel.forward // Successfully computed sum.");
	}
}
