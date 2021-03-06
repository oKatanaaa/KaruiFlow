#include "MulCUDAKernels.h"
#include "../../core/headers/CudaContextManager.h"
#include "../../core/headers/LoggingUtils.h"
#include <spdlog/spdlog.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "device_launch_parameters.h"
#include "../sum/SumCUDAKernels.h"

namespace karuiflow {
	MulCudaKernel::MulCudaKernel(DType* dtype) {
		if (dtype->getName() == "float32") {
			m_CuDtype = CUDA_R_32F;
			m_ComputeType = CUTENSOR_COMPUTE_32F;
			m_Scalar = (void*)&m_FloatScalar;
		}
		else if (dtype->getName() == "int32") {
			m_CuDtype = CUDA_R_32I;
			m_ComputeType = CUTENSOR_COMPUTE_32I;
			m_Scalar = (void*)&m_IntScalar;
		}
		else
			throw std::runtime_error("MulCudaKernel // Expected float32 or int32, but received" + dtype->getName());
	}

	std::vector<int> MulCudaKernel::initTensorDescriptor(cutensorTensorDescriptor_t& desc, Storage* storage) {
		std::vector<int> tensorDim = storage->getShape();
		std::vector<long long> extent = std::vector<long long>(tensorDim.begin(), tensorDim.end());
		std::vector<int> mode;
		std::vector<long long> stride{ 1 };

		// Init stride
		for (int i = tensorDim.size() - 1; i > 0; i--) {
			stride.insert(stride.begin(), { tensorDim[i] * stride.front() });
		}
		// Init mode
		for (int i = 0; i < tensorDim.size(); i++) {
			mode.push_back(i);
		}
		std::vector<long long> updatedExtent;
		std::vector<int> updatedMode;
		std::vector<long long> updatedStride;
		for (int i = 0; i < tensorDim.size(); i++) {
			if (extent[i] == 1)
				continue;
			updatedExtent.push_back(extent[i]);
			updatedMode.push_back(i);
			updatedStride.push_back(stride[i]);
		}

		CUTENSOR_CHECK(cutensorInitTensorDescriptor(
			CudaContextManager::getCuTensorHandle(),
			&desc,
			updatedMode.size(),
			updatedExtent.data(),
			updatedStride.data(),
			m_CuDtype,
			CUTENSOR_OP_IDENTITY
		));
		std::string msg = "Initialized descriptor for a tensor. ";
		std::string modeStr = "\nmode=" + shapeToString(updatedMode);
		std::string extentStr = "\nextent=" + shapeToString(std::vector<int>{ updatedExtent.begin(), updatedExtent.end() });
		std::string strideStr = "\nstride=" + shapeToString(std::vector<int>{ updatedStride.begin(), updatedStride.end() });
		spdlog::debug(msg + modeStr + extentStr + strideStr);
		return updatedMode;
	}

	void MulCudaKernel::forward(std::vector<Storage*> inputs, Storage* output) {
		void* d_A = inputs[0]->getData();
		void* d_B = inputs[1]->getData();
		void* d_C = output->getData();
		cutensorHandle_t* handle = CudaContextManager::getCuTensorHandle();

		// Creating a descriptors for input tensors
		spdlog::debug("MulCudaKernel.forward // Creating descriptors for tensors...");
		cutensorTensorDescriptor_t descA;
		auto modeA = initTensorDescriptor(descA, inputs[0]);

		cutensorTensorDescriptor_t descB;
		auto modeB = initTensorDescriptor(descB, inputs[1]);

		// Creating a descriptor for the output Tensor
		cutensorTensorDescriptor_t descC;
		auto modeC = initTensorDescriptor(descC, output);
		spdlog::debug("MulCudaKernel.forward // Finished creating descriptors.");

		// Perform compute
		spdlog::debug("MulCudaKernel.forward // Computing mul...");
		CUTENSOR_CHECK(cutensorElementwiseBinary(handle,
			(const void*)m_Scalar, d_B, &descB, modeB.data(),
			(const void*)m_Scalar, d_A, &descA, modeA.data(),
			d_C, &descA, modeA.data(),
			CUTENSOR_OP_MUL, m_CuDtype, 0 /* stream */));

		spdlog::debug("MulCudaKernel.forward // Successfully computed mul.");
	}

	void MulCudaKernel::backward(std::vector<Storage*> inputs, std::vector<bool> requiresGrad,
		Storage* outerGradient, std::vector<Storage*> outputGradients) {
		cutensorHandle_t* handle = CudaContextManager::getCuTensorHandle();
		void* d_A = inputs[0]->getData();
		void* d_B = inputs[1]->getData();
		void* d_OuterGrad = outerGradient->getData();

		// Creating descriptors
		spdlog::debug("MulCudaKernel.backward // Creating descriptors for tensors...");
		cutensorTensorDescriptor_t descA;
		auto modeA = initTensorDescriptor(descA, inputs[0]);
		
		cutensorTensorDescriptor_t descB;
		auto modeB = initTensorDescriptor(descB, inputs[1]);
		

		cutensorTensorDescriptor_t descOuterGrad;
		auto modeOuterGrad = initTensorDescriptor(descOuterGrad, outerGradient);

		spdlog::debug("MulCudaKernel.backward // Finished creating descriptors.");

		// Perform compute
		float alpha = 1.f;
		if (requiresGrad[0]) {
			// gradA = B * outerGrad
			void* d_GradA = outputGradients[0]->getData();
			spdlog::debug("MulCudaKernel.backward // Computing gradient for the left tensor...");
			CUTENSOR_CHECK(cutensorElementwiseBinary(handle,
				(const void*)&alpha, d_B, &descB, modeB.data(),
				(const void*)&alpha, d_OuterGrad, &descOuterGrad, modeOuterGrad.data(),
				d_GradA, &descOuterGrad, modeOuterGrad.data(),
				CUTENSOR_OP_MUL, m_CuDtype, 0 /* stream */));
			spdlog::debug("MulCudaKernel.forward // Successfully computed gradient.");
		}

		if (requiresGrad[1]) {
			// gradB = sum(A * outerGrad, dim=broadcastDims(B))
			void* d_GradB = outputGradients[1]->getData();
			spdlog::debug("MulCudaKernel.backward // Computing gradient for the right tensor...");
			Storage* buff = outerGradient->createSimilar();
			CUTENSOR_CHECK(cutensorElementwiseBinary(handle,
				(const void*)&alpha, d_A, &descA, modeA.data(),
				(const void*)&alpha, d_OuterGrad, &descOuterGrad, modeOuterGrad.data(),
				buff->getData(), &descOuterGrad, modeOuterGrad.data(),
				CUTENSOR_OP_MUL, m_CuDtype, 0 /* stream */));

			// Determine which dimensions to reduce
			std::vector<int> dimsToReduce;
			std::vector<int> bGradShape = outputGradients[1]->getShape();
			for (int i = 0; i < bGradShape.size(); i++)
				if (bGradShape[i] == 1)
					dimsToReduce.push_back(i);

			SumCudaKernel sumKernel(dimsToReduce, outputGradients[1]->getDtype());
			sumKernel.forward({ buff }, outputGradients[1]);

			delete buff;
			spdlog::debug("MulCudaKernel.backward // Successfully computed gradient.");
		}
	}

}
