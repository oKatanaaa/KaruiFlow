#include "Mul.h"
#include "MulCPUKernels.h"
#include "MulCUDAKernels.h"
#include "../../core/headers/LoggingUtils.h"


namespace karuiflow {

	std::string Mul::getOpName() {
		return "Mul";
	}

	Kernel* Mul::instantiateKernel(std::vector<TensorSpecs> inputs) {
		Device* device = inputs[0].device;
		if (device->getDeviceName() == "cpu")
			return new MulNumpyKernel();

		if (device->getDeviceName() == "cuda" && inputs[0].dtype->getName() == "float32")
			return new MulCudaKernel(inputs[0].dtype->copy());
		else
			throw std::runtime_error("No cuda kernel for operation " + getOpName() + " supports data with dtype " 
				+ inputs[0].dtype->getName() +
				". Please use float32 data or perform computation on CPU."
			);
	}

	TensorSpecs Mul::inferOutputTensorSpecs(std::vector<TensorSpecs> inputs) {
		Shape nonBroadcastableShape = inputs[0].shape;
		Shape broadcastableShape = inputs[1].shape;
		
		if (nonBroadcastableShape.size() != broadcastableShape.size())
			throw std::runtime_error(getOpName() + " // Tensors must have the same number of dimensions, " +
				"but received tensors with ndims " + std::to_string(nonBroadcastableShape.size()) + " and " +
				std::to_string(broadcastableShape.size()));

		for (int i = 0; i < nonBroadcastableShape.size(); i++) {
			if (nonBroadcastableShape[i] != broadcastableShape[i] && broadcastableShape[i] != 1)
				throw std::runtime_error(std::string("The second tensor is not broadcastable to the first tensor. ") +
					"Received tensors with dimensions " + shapeToString(nonBroadcastableShape) + " and " +
					shapeToString(broadcastableShape) + 
					"\nAn example of broadcastable pair of tensors: [3, 4, 5] and [1, 4, 1]. "
					"\nExamples of wrong pairs: [1, 4, 1] and [3, 4, 5], [3, 4, 5] and [1, 4, 2]"
				);
		}
		return TensorSpecs{ inputs[0].dtype->copy(), inputs[0].shape, inputs[0].device };
	}
}
