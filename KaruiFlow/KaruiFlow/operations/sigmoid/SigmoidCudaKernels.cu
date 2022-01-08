#include "SigmoidCUDAKernels.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include "device_launch_parameters.h"


__device__ __inline__ float sigmoid(float x) {
	return 1.f / (1.f + exp(-x));
}


__global__ void cudaForwardSigmoid(float* inputData, float* outputData, size_t nElems) {
	int tId = blockDim.x * blockIdx.x + threadIdx.x;
	if (tId < nElems) {
		outputData[tId] = sigmoid(inputData[tId]);
	}
}


__global__ void cudaBackwardSigmoid(float* inputData, float* outerGradient, float* outputGradient, size_t nElems) {
	int tId = blockDim.x * blockIdx.x + threadIdx.x;
	if (tId < nElems) {
		outputGradient[tId] = sigmoid(inputData[tId]) * (1 - sigmoid(inputData[tId])) * outerGradient[tId];
	}
}


namespace karuiflow {
	void SigmoidCudaKernel::forward(std::vector<Storage*> inputs, Storage* output) {
		// Cuda kernels are guarantied to receive Storages that store their data
		// on device (cuda device).
		float* inputData = (float*)inputs[0]->getData();
		float* outputData = (float*)output->getData();
		size_t nElems = inputs[0]->getSize();

		int nBlocks = (nElems + m_ThreadsPerBlock - 1) / m_ThreadsPerBlock;

		cudaForwardSigmoid<<<nBlocks, m_ThreadsPerBlock>>>(inputData, outputData, nElems);
	}

	void SigmoidCudaKernel::backward(std::vector<Storage*> inputs, std::vector<bool> requiresGrad,
		Storage* outerGradient, std::vector<Storage*> outputGradients) {
		float* inputData = (float*)inputs[0]->getData();
		float* _outerGradient = (float*)outerGradient->getData();
		float* outputGradient = (float*)outputGradients[0]->getData();
		size_t nElems = inputs[0]->getSize();

		int nBlocks = (nElems + m_ThreadsPerBlock - 1) / m_ThreadsPerBlock;

		if (requiresGrad[0])
			cudaBackwardSigmoid<<<nBlocks, m_ThreadsPerBlock>>>(inputData, _outerGradient, outputGradient, nElems);
	}

}
