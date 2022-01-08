#include "SoftmaxCUDAKernels.h"
#include "../../core/headers/CudaContextManager.h"
#include <cuda_runtime.h>
#include <cudnn.h>
#include "device_launch_parameters.h"


namespace karuiflow {
	void SoftmaxCudaKernel::forward(std::vector<Storage*> inputs, Storage* output) {
		// Cuda kernels are guarantied to receive Storages that store their data
		// on device (cuda device).
		float* inputData = (float*)inputs[0]->getData();
		float* outputData = (float*)output->getData();
		cudnnHandle_t* handle = CudaContextManager::getCudnnHandle();

		std::vector<int> inputDataShape = inputs[0]->getShape();
		// For 2D Tensor: n=batch-size, c=channels
		int n = inputDataShape[0]; 
		int c = inputDataShape[1];
		int h = 1, w = 1;
		cudnnTensorDescriptor_t inputDesc;
		cudnnCreateTensorDescriptor(&inputDesc);
		cudnnSetTensor4dDescriptor(inputDesc,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			n, c, h, w);

		cudnnTensorDescriptor_t outputDesc;
		cudnnCreateTensorDescriptor(&outputDesc);
		cudnnSetTensor4dDescriptor(outputDesc,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			n, c, h, w);

		const float alpha = 1, beta = 0;
		// Straightforward softmax operation is computed 
		// per spatial location (H,W) per image (N) across dimension C.
		cudnnSoftmaxForward(*handle,
			CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_MODE_CHANNEL,
			&alpha, inputDesc, inputData,
			&beta,outputDesc, outputData);
	}

	void SoftmaxCudaKernel::backward(std::vector<Storage*> inputs, std::vector<bool> requiresGrad,
		Storage* outerGradient, std::vector<Storage*> outputGradients) {
		if (requiresGrad[0]) {
			cudnnHandle_t* handle = CudaContextManager::getCudnnHandle();
			std::vector<int> inputDataShape = inputs[0]->getShape();
			// For 2D Tensor: n=batch-size, c=channels
			int n = inputDataShape[0];
			int c = inputDataShape[1];
			int h = 1, w = 1;
			cudnnTensorDescriptor_t inputDesc;
			cudnnCreateTensorDescriptor(&inputDesc);
			cudnnSetTensor4dDescriptor(inputDesc,
				CUDNN_TENSOR_NCHW,
				CUDNN_DATA_FLOAT,
				n, c, h, w);

			cudnnTensorDescriptor_t outputDesc;
			cudnnCreateTensorDescriptor(&outputDesc);
			cudnnSetTensor4dDescriptor(outputDesc,
				CUDNN_TENSOR_NCHW,
				CUDNN_DATA_FLOAT,
				n, c, h, w);

			cudnnTensorDescriptor_t gradDesc;
			cudnnCreateTensorDescriptor(&gradDesc);
			cudnnSetTensor4dDescriptor(gradDesc,
				CUDNN_TENSOR_NCHW,
				CUDNN_DATA_FLOAT,
				n, c, h, w);

			forward(inputs, outputGradients[0]);
			float* _outerGradient = (float*)outerGradient->getData();
			float* outputGradient = (float*)outputGradients[0]->getData();
			const float alpha = 1, beta = 0;
	
			cudnnSoftmaxBackward(*handle,
				CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_MODE_CHANNEL,
				&alpha, inputDesc, outputGradient,
				gradDesc, _outerGradient,
				&beta, outputDesc, outputGradient);
		}
		
	}
}
