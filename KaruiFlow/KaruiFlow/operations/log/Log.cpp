#include "Log.h"
#include "LogCPUKernels.h"
#include "LogCUDAKernels.h"


namespace karuiflow {

	std::string Log::getOpName() {
		return "Log";
	}

	Kernel* Log::instantiateKernel(std::vector<TensorSpecs> inputs) {
		Device* device = inputs[0].device;
		if (device->getDeviceName() == "cpu")
			return new LogNumpyKernel();

		std::string dtype = inputs[0].dtype->getName();
		if (device->getDeviceName() == "cuda" && dtype == "float32")
			return new LogCudaKernel();
		else
			throw std::runtime_error("No kernel for dtype " + dtype + " is available for operation" + getOpName());
	}

	TensorSpecs Log::inferOutputTensorSpecs(std::vector<TensorSpecs> inputs) {
		return TensorSpecs{ inputs[0].dtype->copy(), inputs[0].shape, inputs[0].device };
	}
}
