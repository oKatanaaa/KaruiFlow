#include "MatMulCUDAKernels.h"
#include "../../core/headers/CudaContextManager.h"
#include "../../core/headers/LoggingUtils.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "device_launch_parameters.h"


namespace karuiflow {
	/*
	* Multiplies C-stored matrices: C = A * B. 
	* For the theory behind the code please refer to https://peterwittek.com/cublas-matrix-c-style.html
	* 
	* @param a
	* Matrix A.
	* @param transposeA
	* Whether to transpose matrix A.
	* @param b
	* Matrix B.
	* @param transposeB
	* Whether to transpose matrix B.
	* @param c
	* Matrix to store the multiplication result in.
	*/
	void cublasMatmul(Storage* a, bool transposeA, Storage* b, bool transposeB, Storage* c) {
		float* A = (float*)a->getData();
		float* B = (float*)b->getData();
		float* C = (float*)c->getData();

		cublasOperation_t opA;
		if (transposeA)
			opA = CUBLAS_OP_T;
		else
			opA = CUBLAS_OP_N;

		cublasOperation_t opB;
		if (transposeB)
			opB = CUBLAS_OP_T;
		else
			opB = CUBLAS_OP_N;

		cublasHandle_t* handle = CudaContextManager::getCublasHandle();
		//4. A[m, k]*B[k, n] = C[m, n]
		// cublas considers matrices to be stored in column-major format:
		// ~ B[n,k]*A[k,m] = C[n,m]
		std::vector<int> shapeA = a->getShape();
		std::vector<int> shapeB = b->getShape();
		std::vector<int> shapeC = c->getShape();
		int colB = !transposeB ? shapeB[1] : shapeB[0]; //m
		int rowA = !transposeA ? shapeA[0] : shapeA[1]; //n
		int colA = !transposeA ? shapeA[1] : shapeA[0]; //k
		int ldA = shapeA[1];
		int ldB = shapeB[1];
		int ldC = shapeC[1];
		const float alpha = 1.f;
		const float beta = 0.f;
		// lda=ldc=colB, ldb=colA
		//  ..., m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
		CUBLAS_CHECK(cublasSgemm(
			*handle, 
			opB, opA,
			colB, rowA, colA, 
			&alpha, B, ldB,
			A, ldA, &beta,
			C, ldC));
	}

	void MatMulCudaKernel::forward(std::vector<Storage*> inputs, Storage* output) {
		cublasMatmul(inputs[0], false, inputs[1], false, output);
	}

	void MatMulCudaKernel::backward(std::vector<Storage*> inputs, std::vector<bool> requiresGrad,
		Storage* outerGradient, std::vector<Storage*> outputGradients) {
		bool aRequiresGrad = requiresGrad[0];
		bool bRequiresGrad = requiresGrad[1];
		if (aRequiresGrad) {
			cublasMatmul(outerGradient, false, inputs[1], true, outputGradients[0]);
		}
		if (bRequiresGrad) {
			cublasMatmul(inputs[0], true, outerGradient, false, outputGradients[1]);
		}
	}
}
