#pragma once
#include "Matmul.h"
#include "MatMulCPUKernels.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "device_launch_parameters.h"
#include "../../core/headers/Kernel.h"
#include "../../core/headers/CudaContextManager.h"
#include "../../core/headers/LoggingUtils.h"



namespace karuiflow {
	class MatMulCudaKernel : public Kernel {
	public:
		MatMulCudaKernel() {};
		void forward(std::vector<Storage*> inputs, Storage* output) override;
		void backward(std::vector<Storage*> inputs, std::vector<bool> requiresGrad,
			Storage* outerGradient, std::vector<Storage*> outputGradients) override;

	private:
		int m_ThreadsPerBlock = 256;
	};
}

namespace karuiflow {
	void MatMulCudaKernel::forward(std::vector<Storage*> inputs, Storage* output) {
		// Cuda kernels are guarantied to receive Storages that store their data
		// on device (cuda device).
		float* inputA = (float*)inputs[0]->getData();
		float* inputB = (float*)inputs[1]->getData();
		float* outputC = (float*)output->getData();

		std::string dtypeA = inputs[0]->getDtype()->getName();
		if (dtypeA != "float32") {
			throw std::runtime_error("MetMulCudaKernel.forward // Expected float32, but received" + dtypeA);
		}

		//4. A[m, k]*B[k, n] = C[m, n]
		// cublas consider that matrices stored incolumn-major format:
		// ~ B[n,k]*A[k,m] = C[n,m]
		Shape shapeA = inputs[0]->getShape();
		Shape shapeB = inputs[1]->getShape();
		cublasHandle_t* handle = CudaContextManager::getCublasHandle();
		int colB = shapeB[1]; //m
		int rowA = shapeA[0]; //n
		int colA = shapeA[1]; //k
		const float alpha = 1;
		const float beta = 0;
		// lda=ldc=colB, ldb=colA
		//  ..., m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
		cublasSgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_N, colB, rowA, colA, &alpha, inputB, colB, inputA, colA, &beta, outputC, colB);
	}

	void MatMulCudaKernel::backward(std::vector<Storage*> inputs, std::vector<bool> requiresGrad,
		Storage* outerGradient, std::vector<Storage*> outputGradients) {
		float* inputA = (float*)inputs[0]->getData();
		float* inputB = (float*)inputs[1]->getData();
		float* _outerGradient = (float*)outerGradient->getData();
		float* A_outputGradient = (float*)outputGradients[0]->getData();
		float* B_outputGradient = (float*)outputGradients[1]->getData();
		bool A_requires_grad = requiresGrad[0];
		bool B_requires_grad = requiresGrad[1];
		Shape shapeA = inputs[0]->getShape();
		Shape shapeB = inputs[1]->getShape();
		Shape shapeG = outerGradient->getShape();
		cublasHandle_t* handle = CudaContextManager::getCublasHandle();

		int colB = shapeB[0]; // transponsed
		int colA = shapeA[0];  // transponsed
		int colG = shapeG[1];
		int rowG = shapeG[0];
		const float alpha = 1;
		const float beta = 0;

		if (A_requires_grad) {
			cublasSgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_T, colB, rowG, colG, &alpha, inputB, colB, _outerGradient, colG, &beta, A_outputGradient, colB);
		}

		if (B_requires_grad) {
			cublasSgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_T, colA, rowG, colG, &alpha, inputA, colA, _outerGradient, colG, &beta, B_outputGradient, colA);
		}
	}
}