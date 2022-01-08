#include "Sigmoid.h"
#include "SigmoidCPUKernels.h"
#include "SigmoidCUDAKernels.h"
#include "../../core/headers/memory/DType.h"


namespace karuiflow {

	std::string Sigmoid::getOpName() {
		return "Sigmoid";
	}

	Kernel* Sigmoid::instantiateKernel(std::vector<TensorSpecs> inputs) {
		Device* device = inputs[0].device;
		if (device->getDeviceName() == "cpu")
			return new SigmoidNumpyKernel();

		std::string dtype = inputs[0].dtype->getName();
		if (device->getDeviceName() == "cuda" && dtype == "float32")
			return new SigmoidCudaKernel();
		else
			throw std::runtime_error("No kernel for dtype " + dtype + " is available for operation" + getOpName());
	}

	TensorSpecs Sigmoid::inferOutputTensorSpecs(std::vector<TensorSpecs> inputs) {
		return TensorSpecs{ new Float32(), inputs[0].shape, inputs[0].device };
	}
}