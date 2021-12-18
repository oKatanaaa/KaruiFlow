#include "Softmax.h"
#include "SoftmaxCPUKernels.h"


namespace karuiflow {

	std::string Softmax::getOpName() {
		return "Softmax";
	}

	Kernel* Softmax::instantiateKernel(std::vector<TensorSpecs> inputs) {
		Device* device = inputs[0].device;
		if (device->getDeviceName() == "cpu")
			return new SoftmaxNumpyKernel();

		if (device->getDeviceName() == "cuda")
			throw KF_ERROR(std::runtime_error("Cuda is not supported."));
	}

	TensorSpecs Softmax::inferOutputTensorSpecs(std::vector<TensorSpecs> inputs) {
		TensorSpecs specs = inputs[0];
		if (specs.shape.size() != 2)
			throw KF_ERROR(UnsuppotedShapes(getOpName(), { specs.shape }));

		return TensorSpecs{ inputs[0].dtype, inputs[0].shape, inputs[0].device };
	}
}