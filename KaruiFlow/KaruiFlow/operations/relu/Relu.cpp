#include "Relu.h"
#include "ReluCPUKernels.h"
#include "ReluCUDAKernels.h"


namespace karuiflow {

	std::string Relu::getOpName() {
		return "Relu";
	}

	Kernel* Relu::instantiateKernel(std::vector<TensorSpecs> inputs) {
		Device* device = inputs[0].device;
		if (device->getDeviceName() == "cpu")
			return new ReluNumpyKernel();

		if (device->getDeviceName() == "cuda") {
			std::string dtypeName = inputs[0].dtype->getName();
			if (dtypeName == "float32")
				return new ReluCudaKernel<float>();
			else if (dtypeName == "int32")
				return new ReluCudaKernel<int>();
			else
				throw std::runtime_error("Expected int32 or float32, but received " + dtypeName);
		}
	}

	TensorSpecs Relu::inferOutputTensorSpecs(std::vector<TensorSpecs> inputs) {
		return TensorSpecs{ inputs[0].dtype->copy(), inputs[0].shape, inputs[0].device };
	}
}