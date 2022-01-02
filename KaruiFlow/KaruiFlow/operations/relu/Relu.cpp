#include "Relu.h"
#include "ReluCPUKernels.h"


namespace karuiflow {

	std::string Relu::getOpName() {
		return "Relu";
	}

	Kernel* Relu::instantiateKernel(std::vector<TensorSpecs> inputs) {
		Device* device = inputs[0].device;
		if (device->getDeviceName() == "cpu")
			return new ReluNumpyKernel();

		if (device->getDeviceName() == "cuda")
			throw KF_ERROR(std::runtime_error("Cuda is not supported."));
	}

	TensorSpecs Relu::inferOutputTensorSpecs(std::vector<TensorSpecs> inputs) {
		return TensorSpecs{ inputs[0].dtype->copy(), inputs[0].shape, inputs[0].device };
	}
}