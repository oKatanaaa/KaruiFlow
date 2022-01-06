#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include "device_launch_parameters.h"
#include "../../core/headers/Kernel.h"


namespace karuiflow {
	template<class T>
	class SigmoidCudaKernel : public Kernel {
	public:
		SigmoidCudaKernel() {};
		void forward(std::vector<Storage*> inputs, Storage* output) override;
		void backward(std::vector<Storage*> inputs, std::vector<bool> requiresGrad,
			Storage* outerGradient, std::vector<Storage*> outputGradients) override;

	private:
		int m_ThreadsPerBlock = 256;
	};
}


__device__ float sigmoid(float x) {
	return 1.0f / (1 + exp(-x));
}

template<typename T>
__global__ void cudaForwardSigmoid(T* inputData, T* outputData, size_t nElems) {
	int tId = blockDim.x * blockIdx.x + threadIdx.x;
	if (tId < nElems) {
		outputData[tId] = sigmoid(inputData[tId]);
	}
}

template<typename T>
__global__ void cudaBackwardSigmoid(T* inputData, T* outerGradient, T* outputGradient, size_t nElems) {
	int tId = blockDim.x * blockIdx.x + threadIdx.x;
	if (tId < nElems) {
		outputGradient[tId] = sigmoid(outerGradient[tId])*(1 - sigmoid(outerGradient[tId]))*outerGradient[tId];
	}
}


namespace karuiflow {
	template <class T>
	void SigmoidCudaKernel<T>::forward(std::vector<Storage*> inputs, Storage* output) {
		// Cuda kernels are guarantied to receive Storages that store their data
		// on device (cuda device).
		T* inputData = (T*)inputs[0]->getData();
		T* outputData = (T*)output->getData();
		size_t nElems = inputs[0]->getSize();

		int nBlocks = (nElems + m_ThreadsPerBlock - 1) / m_ThreadsPerBlock;

		cudaForwarSigmoid<T><<<nBlocks, m_ThreadsPerBlock>>>(inputData, outputData, nElems);
	}

	template <class T>
	void SigmoidCudaKernel<T>::backward(std::vector<Storage*> inputs, std::vector<bool> requiresGrad,
		Storage* outerGradient, std::vector<Storage*> outputGradients) {
		T* inputData = (T*)inputs[0]->getData();
		T* _outerGradient = (T*)outerGradient->getData();
		T* outputGradient = (T*)outputGradients[0]->getData();
		size_t nElems = inputs[0]->getSize();

		int nBlocks = (nElems + m_ThreadsPerBlock - 1) / m_ThreadsPerBlock;

		if (requiresGrad[0])
			cudaBackwardSigmoid<T><<<nBlocks, m_ThreadsPerBlock>>>(inputData, _outerGradient, outputGradient, nElems);
	}

}
