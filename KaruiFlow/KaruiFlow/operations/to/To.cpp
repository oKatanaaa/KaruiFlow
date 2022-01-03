#include "To.h"
#include "ToKernel.h"

namespace karuiflow {
	To::To(Device* device) {
		if (device == nullptr)
			throw std::runtime_error(getOpName() + " // Received nullptr as device.");
		m_Device = device;
	}

	std::string To::getOpName()  { return "To"; };

	Kernel* To::instantiateKernel(std::vector<TensorSpecs> inputs) {
		return new ToKernel();
	}

	TensorSpecs To::inferOutputTensorSpecs(std::vector<TensorSpecs> inputs) {
		TensorSpecs specs = TensorSpecs{ inputs[0].dtype->copy(), inputs[0].shape, m_Device };
		return specs;
	}
}