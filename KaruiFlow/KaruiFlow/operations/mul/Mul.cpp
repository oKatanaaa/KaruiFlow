#include "Mul.h"
#include "MulCPUKernels.h"
#include "../../core/headers/Op.h"


namespace karuiflow {

	std::string Mul::getOpName() {
		return "Mul";
	}

	Kernel* Mul::instantiateKernel(std::vector<TensorSpecs> inputs) {
		Device* device = inputs[0].device;
		if (device->getDeviceName() == "cpu")
			return new MulNumpyKernel();

		if (device->getDeviceName() == "cuda")
			throw KF_ERROR(std::runtime_error("Cuda is not supported."));
	}

	TensorSpecs Mul::inferOutputTensorSpecs(std::vector<TensorSpecs> inputs) {
		Shape newShape;
		if (inputs[0].shape.size() == 0 || (inputs[0].shape.size() == 1 && inputs[0].shape[0] == 1))
			// Multiplying with a scalar
			newShape = inputs[1].shape;
		else if (inputs[1].shape.size() == 0 || (inputs[1].shape.size() == 1 && inputs[1].shape[0] == 1))
			// Multiplying with a scalar
			newShape = inputs[0].shape;
		else if (inputs[0].shape.size() == inputs[1].shape.size()) {
			for (int i = 0; i < inputs[0].shape.size(); i++) {
				int maxDim = std::max(inputs[0].shape[i], inputs[1].shape[i]);
				newShape.push_back(maxDim);
			}
		}
		else
			throw KF_ERROR(InconsistentShapes(getOpName(), { inputs[0].shape, inputs[1].shape }));

		return TensorSpecs{ inputs[0].dtype->copy(), newShape, inputs[0].device };
	}
}
