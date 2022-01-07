#include "To.h"
#include "ToKernel.h"
#include <spdlog/spdlog.h>

namespace karuiflow {
	To::To(Device* device) {
		spdlog::debug("Creating To operation.");
		if (device == nullptr)
			throw std::runtime_error(getOpName() + " // Received nullptr as device.");
		m_Device = device;
		spdlog::debug("Successfully created To operation.");
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