#include "Add.h"
#include "AddCPUKernels.h"


namespace karuiflow {

	std::string Add::getOpName() {
		return "Add";
	}

	Kernel* Add::instantiateKernel(std::vector<TensorSpecs> inputs) {
		Device* device = inputs[0].device;
		if (device->getDeviceName() == "cpu")
			return new AddNumpyKernel();

		if (device->getDeviceName() == "cuda")
			throw KF_ERROR(std::runtime_error("Cuda is not supported."));
	}

	TensorSpecs Add::inferOutputTensorSpecs(std::vector<TensorSpecs> inputs) {
		return TensorSpecs{ inputs[0].dtype->copy(), inputs[0].shape, inputs[0].device };
	}
}
