#include "SumCUDAKernels.h"
#include "../../core/headers/CudaContextManager.h"
#include "../../core/headers/LoggingUtils.h"
#include <spdlog/spdlog.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "device_launch_parameters.h"


namespace karuiflow {
	SumCudaKernel::SumCudaKernel(std::vector<int> dim, DType* dtype) {
		m_Dim = dim;
		if (dtype->getName() == "float32") {
			m_CuDtype = CUDA_R_32F;
			m_ComputeType = CUTENSOR_COMPUTE_32F;
		}
		else if (dtype->getName() == "int32") {
			m_CuDtype = CUDA_R_32I;
			m_ComputeType = CUTENSOR_COMPUTE_32I;
		}
		else
			throw std::runtime_error("SumCudaKernel // Expected float32 or int32, but received" + dtype->getName());
	}

	std::vector<int> SumCudaKernel::initTensorDescriptor(cutensorTensorDescriptor_t& desc, Storage* storage, Storage* notReduced) {
		std::vector<int> tensorDim = storage->getShape();
		std::vector<long long> extent;
		std::vector<int> mode;
		std::vector<long long> stride{ 1 };

		// Init extent
		if (notReduced == nullptr)
			extent = std::vector<long long>(tensorDim.begin(), tensorDim.end());
		// Init stride
		for (int i = tensorDim.size() - 1; i > 0; i--) {
			stride.insert(stride.begin(), { tensorDim[i] * stride.front() });
		}
		// Init mode
		if (notReduced == nullptr)
			for (int i = 0; i < tensorDim.size(); i++) {
				mode.push_back(i);
			}
		else
			for (int i = 0; i < notReduced->getShape().size(); i++) {
				if (std::find(m_Dim.begin(), m_Dim.end(), i) == m_Dim.end()) {
					mode.push_back(i);
					extent.push_back(notReduced->getShape()[i]);
				}
			}
		CUTENSOR_CHECK(cutensorInitTensorDescriptor(
			CudaContextManager::getCuTensorHandle(),
			&desc,
			mode.size(),
			extent.data(),
			stride.data(),
			m_CuDtype,
			CUTENSOR_OP_IDENTITY
		));
		std::string msg;
		if (notReduced == nullptr)
			msg = "Initialized descriptor for not reduced tensor. ";
		else
			msg = "Initialized descriptor for reduced tensor. ";

		std::string modeStr = "\nmode=" + shapeToString(mode);
		std::string extentStr = "\nextent=" + shapeToString(std::vector<int>{ extent.begin(), extent.end() });
		std::string strideStr = "\nstride=" + shapeToString(std::vector<int>{ stride.begin(), stride.end() });
		spdlog::debug(msg + modeStr + extentStr + strideStr);
		return mode;
	}

	void SumCudaKernel::forward(std::vector<Storage*> inputs, Storage* output) {
		void* d_A = inputs[0]->getData();
		void* d_C = output->getData();
		cutensorHandle_t* handle = CudaContextManager::getCuTensorHandle();

		// Creating a descriptor for the Tensor to reduce
		spdlog::debug("SumCudaKernel.forward // Creating descriptors for tensors...");
		cutensorTensorDescriptor_t descA;
		auto modeA = initTensorDescriptor(descA, inputs[0], nullptr);

		// Creating a descriptor for the output Tensor
		cutensorTensorDescriptor_t descC;
		auto modeC = initTensorDescriptor(descC, output, inputs[0]);
		spdlog::debug("SumCudaKernel.forward // Finished creating descriptors.");

		// Querry workspace
		spdlog::debug("SumCudaKernel.forward // Quering workspace...");
		const cutensorOperator_t opReduce = CUTENSOR_OP_ADD;
		uint64_t worksize = 0;
		CUTENSOR_CHECK(cutensorReductionGetWorkspace(handle,
			d_A, &descA, modeA.data(),
			d_C, &descC, modeC.data(),
			d_C, &descC, modeC.data(),
			opReduce, m_ComputeType, &worksize));

		void* work = nullptr;
		
		if (worksize > 0)
			output->getDevice()->allocateMemory(&work, worksize);
		spdlog::debug("SumCudaKernel.forward // Workspace acquired.");

		// Perform compute
		spdlog::debug("SumCudaKernel.forward // Computing sum...");
		float alpha = 1.f;
		float beta = 0.f;
		CUTENSOR_CHECK(cutensorReduction(handle,
			(const void*)&alpha, d_A, &descA, modeA.data(),
			(const void*)&beta,  d_C, &descC, modeC.data(),
								 d_C, &descC, modeC.data(),
			opReduce, m_ComputeType, work, worksize, 0 /* stream */));

		if (work != nullptr)
			output->getDevice()->deallocateMemory(work);

		spdlog::debug("SumCudaKernel.forward // Successfully computed sum.");
	}

	template<typename T>
	__global__ void setOnes(T* x, size_t n) {
		size_t tId = blockDim.x * blockIdx.x + threadIdx.x;
		x[tId] = (T)1 * (T)(tId < n);
	}

	void SumCudaKernel::backward(std::vector<Storage*> inputs, std::vector<bool> requiresGrad,
		Storage* outerGradient, std::vector<Storage*> outputGradients) {
		/*
		* Gradient of any entry in the sum is equal to one. So we first set every entry
		* in the gradient to one. After that we perform multiplication with the outer gradient.
		* The overall formula for the gradient looks like this: outputGrad = outerGrad * onesLike(input)
		* 
		* It should be noted, that the `outerGrad` tensor has lower rank than the `input`. For example:
		* outerGrad = [1, 2, 3]
		* onesLike(input) = [
		*	[1, 1, 1, 1],
		*	[1, 1, 1, 1],
		*	[1, 1, 1, 1]
		* ]
		* 
		* In this case the `input` tensor was reduced along the second dimension (summed columns).
		* In order to multiply `outerGrad` with `onesLike(input)`, the `outerGrad` must be broadcasted
		* in the following way:
		* broadcast(outerGrad) = [
		*	[1, 1, 1, 1],
		*	[2, 2, 2, 2],
		*	[3, 3, 3, 3]
		* ]
		* Ones `outerGrad` is broadcasted, it can be multiplied with `onesLike(input)` elementwise.
		* The result will be the gradient of the `input` tensor.
		*/
		const int nThreadsPerBlock = 256;
		int nBlocks = (outputGradients[0]->getSize() + nThreadsPerBlock - 1) / nThreadsPerBlock;
		
		switch (m_CuDtype)
		{
		case CUDA_R_32F:
			setOnes<float> << <nBlocks, nThreadsPerBlock >> > ((float*)outputGradients[0]->getData(), outputGradients[0]->getSize());
			break;
		case CUDA_R_32I:
			setOnes<int> << <nBlocks, nThreadsPerBlock >> > ((int*)outputGradients[0]->getData(), outputGradients[0]->getSize());
			break;
		default:
			// Impossible case
			throw std::runtime_error("SumCudaKernel.backward // Expected CUDA_R_32F or CUDA_R_32I, but received " + std::to_string(m_CuDtype));
		}
		
		void* d_A = outputGradients[0]->getData();
		void* d_B = outerGradient->getData();
		cutensorHandle_t* handle = CudaContextManager::getCuTensorHandle();

		/******************************************************
		* Creating a descriptor for the output gradient (d_A) *
		* ****************************************************/
		spdlog::debug("SumCudaKernel.backward // Creating descriptors for tensors...");
		cutensorTensorDescriptor_t descA;
		auto modeA = initTensorDescriptor(descA, outputGradients[0], nullptr);

		/*****************************************************
		* Creating a descriptor for the outer gradient (d_B) *
		* ***************************************************/
		cutensorTensorDescriptor_t descB;
		auto modeB = initTensorDescriptor(descB, outerGradient, outputGradients[0]);

		spdlog::debug("SumCudaKernel.forward // Finished creating descriptors.");

		// Perform compute
		float alpha = 1.f;
		float gamma = 1.f;
		spdlog::debug("SumCudaKernel.forward // Computing gradient...");
		CUTENSOR_CHECK(cutensorElementwiseBinary(handle,
			(const void*)&alpha, d_B, &descB, modeB.data(),
			(const void*)&gamma, d_A, &descA, modeA.data(),
								 d_A, &descA, modeA.data(),
			CUTENSOR_OP_MUL, m_CuDtype, 0 /* stream */));

		spdlog::debug("SumCudaKernel.forward // Successfully computed sum.");
	}
}
