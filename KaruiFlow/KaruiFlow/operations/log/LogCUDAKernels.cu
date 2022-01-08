#include "LogCUDAKernels.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include "device_launch_parameters.h"



__global__ void cudaForwardLog(float* inputData, float* outputData, size_t nElems) {
	int tId = blockDim.x * blockIdx.x + threadIdx.x;
	if (tId < nElems) {
		outputData[tId] = log(inputData[tId]);
	}
}


__global__ void cudaBackwardLog(float* inputData, float* outerGradient, float* outputGradient, size_t nElems) {
	int tId = blockDim.x * blockIdx.x + threadIdx.x;
	if (tId < nElems) {
		outputGradient[tId] = 1.f / (inputData[tId] + 1e-6) * outerGradient[tId];
	}
}


namespace karuiflow {
	
	void LogCudaKernel::forward(std::vector<Storage*> inputs, Storage* output) {
		// Cuda kernels are guarantied to receive Storages that store their data
		// on device (cuda device).
		float* inputData = (float*)inputs[0]->getData();
		float* outputData = (float*)output->getData();
		size_t nElems = inputs[0]->getSize();

		int nBlocks = (nElems + m_ThreadsPerBlock - 1) / m_ThreadsPerBlock;

		cudaForwardLog<<<nBlocks, m_ThreadsPerBlock>>>(inputData, outputData, nElems);
	}

	
	void LogCudaKernel::backward(std::vector<Storage*> inputs, std::vector<bool> requiresGrad,
		Storage* outerGradient, std::vector<Storage*> outputGradients) {
		float* inputData = (float*)inputs[0]->getData();
		float* _outerGradient = (float*)outerGradient->getData();
		float* outputGradient = (float*)outputGradients[0]->getData();
		size_t nElems = inputs[0]->getSize();

		int nBlocks = (nElems + m_ThreadsPerBlock - 1) / m_ThreadsPerBlock;

		if (requiresGrad[0])
			cudaBackwardLog<<<nBlocks, m_ThreadsPerBlock>>>(inputData, _outerGradient, outputGradient, nElems);
	}

}
