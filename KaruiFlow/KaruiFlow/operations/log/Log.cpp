#include "Log.h"
#include "LogCPUKernels.h"


namespace karuiflow {

	std::string Log::getOpName() {
		return "Log";
	}

	Kernel* Log::instantiateKernel(std::vector<TensorSpecs> inputs) {
		Device* device = inputs[0].device;
		if (device->getDeviceName() == "cpu")
			return new LogNumpyKernel();

		if (device->getDeviceName() == "cuda")
			throw KF_ERROR(std::runtime_error("Cuda is not supported."));
	}

	TensorSpecs Log::inferOutputTensorSpecs(std::vector<TensorSpecs> inputs) {
		return TensorSpecs{ inputs[0].dtype->copy(), inputs[0].shape, inputs[0].device };
	}
}
