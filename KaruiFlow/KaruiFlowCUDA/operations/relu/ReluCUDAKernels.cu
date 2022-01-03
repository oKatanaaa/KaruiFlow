#include "operations/relu/ReluCUDAKernels.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include "device_launch_parameters.h"

template<typename T>
__global__ void cudaForwardRelu(T* inputData, T* outputData, size_t nElems) {
	int tId = blockDim.x * blockIdx.x + threadIdx.x;
	if (tId < nElems) {
		outputData[tId] = inputData[tId] * (T)(inputData[tId] > 0);
	}
}


template<typename T>
__global__ void cudaBackwardRelu(T* inputData, T* outerGradient, T* outputGradient, size_t nElems) {
	int tId = blockDim.x * blockIdx.x + threadIdx.x;
	// Equals 1 if input > 0, otherwise equals 0.
	T isGreaterZero = 0;
	if (tId < nElems) {
		isGreaterZero = (T) (inputData[tId] > 0);
		outputGradient[tId] = isGreaterZero * outerGradient[tId];
	}
}


namespace karuiflow {
	template <class T>
	void ReluCudaKernel<T>::forward(std::vector<Storage*> inputs, Storage* output) {
		// Cuda kernels are guarantied to receive Storages that store their data
		// on device (cuda device).
		T* inputData = inputs[0]->getData();
		T* outputData = output->getData();
		size_t nElems = inputs[0]->getSize();

		int nBlocks = (nElems + m_ThreadsPerBlock - 1) / m_ThreadsPerBlock;

		cudaForwardRelu<T><<<nBlocks, nThreadsPerBlock>>> (inputData, outputData, nElems);
	}

	template <class T>
	void ReluCudaKernel<T>::backward(std::vector<Storage*> inputs, std::vector<bool> requiresGrad,
		Storage* outerGradient, std::vector<Storage*> outputGradients) {
		T* inputData = inputs[0]->getData();
		T* _outerGradient = outerGradient->getData();
		T* outputGradient = outputGradients[0]->getData();
		size_t nElems = inputs[0]->getSize();

		int nBlocks = (nElems + m_ThreadsPerBlock - 1) / m_ThreadsPerBlock;

		if (requiresGrad[0])
			cudaBackwardRelu<T><<<nBlocks, m_ThreadsPerBlock>>>(inputData, _outerGradient, outputGradient, nElems);
	}


	void ReluFloatCudaKernel::forward(std::vector<Storage*> inputs, Storage* output) {
		// Cuda kernels are guarantied to receive Storages that store their data
		// on device (cuda device).
		float* inputData = (float*)inputs[0]->getData();
		float* outputData = (float*)output->getData();
		size_t nElems = inputs[0]->getSize();

		int nBlocks = (nElems + m_ThreadsPerBlock - 1) / m_ThreadsPerBlock;

		cudaForwardRelu<float> <<<nBlocks, m_ThreadsPerBlock >> > (inputData, outputData, nElems);
	}

	void ReluFloatCudaKernel::backward(std::vector<Storage*> inputs, std::vector<bool> requiresGrad,
		Storage* outerGradient, std::vector<Storage*> outputGradients) {
		float* inputData = (float*)inputs[0]->getData();
		float* _outerGradient = (float*)outerGradient->getData();
		float* outputGradient = (float*)outputGradients[0]->getData();
		size_t nElems = inputs[0]->getSize();

		int nBlocks = (nElems + m_ThreadsPerBlock - 1) / m_ThreadsPerBlock;

		if (requiresGrad[0])
			cudaBackwardRelu<float> <<<nBlocks, m_ThreadsPerBlock >> > (inputData, _outerGradient, outputGradient, nElems);
	}
}
