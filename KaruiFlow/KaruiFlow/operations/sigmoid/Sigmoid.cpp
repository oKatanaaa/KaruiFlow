#include "Sigmoid.h"
#include "SigmoidCPUKernels.h"


namespace karuiflow {

	std::string Sigmoid::getOpName() {
		return "Sigmoid";
	}

	Kernel* Sigmoid::instantiateKernel(std::vector<TensorSpecs> inputs) {
		Device* device = inputs[0].device;
		if (device->getDeviceName() == "cpu")
			return new SigmoidNumpyKernel();

		if (device->getDeviceName() == "cuda")
			throw KF_ERROR(std::runtime_error("Cuda is not supported."));
	}

	TensorSpecs Sigmoid::inferOutputTensorSpecs(std::vector<TensorSpecs> inputs) {
		return TensorSpecs{ inputs[0].dtype->copy(), inputs[0].shape, inputs[0].device };
	}
}