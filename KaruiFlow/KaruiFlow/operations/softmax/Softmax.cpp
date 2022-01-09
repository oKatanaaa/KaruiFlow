#include "Softmax.h"
#include "SoftmaxCPUKernels.h"
#include "SoftmaxCUDAKernels.h"
#include "../../core/headers/memory/DType.h"

namespace karuiflow {

	std::string Softmax::getOpName() {
		return "Softmax";
	}

	Kernel* Softmax::instantiateKernel(std::vector<TensorSpecs> inputs) {
		Device* device = inputs[0].device;
		if (device->getDeviceName() == "cpu")
			return new SoftmaxNumpyKernel();

		std::string dtype = inputs[0].dtype->getName();
		if (device->getDeviceName() == "cuda" && dtype == "float32")
			return new SoftmaxCudaKernel();
		else
			throw std::runtime_error("No kernel for dtype " + dtype + " is available for operation" + getOpName());

	}

	TensorSpecs Softmax::inferOutputTensorSpecs(std::vector<TensorSpecs> inputs) {
		TensorSpecs specs = inputs[0];
		if (specs.shape.size() != 2)
			throw KF_ERROR(UnsuppotedShapes(getOpName(), { specs.shape }));

		return TensorSpecs{ inputs[0].dtype->copy(), inputs[0].shape, inputs[0].device };
	}
}