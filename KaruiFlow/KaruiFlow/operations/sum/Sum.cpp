#include "Sum.h"
#include "SumCPUKernels.h"


namespace karuiflow {

	std::string Sum::getOpName() {
		return "Sum";
	}

	Kernel* Sum::instantiateKernel(std::vector<TensorSpecs> inputs) {
		Device* device = inputs[0].device;
		if (device->getDeviceName() == "cpu")
			return new SumNumpyKernel(m_Dim);

		if (device->getDeviceName() == "cuda")
			throw KF_ERROR(std::runtime_error("Cuda is not supported."));
	}

	bool hasDim(int dim, std::vector<int>& dims) {
		for (int _dim : dims)
			if (dim == _dim)
				return true;
		return false;
	}

	TensorSpecs Sum::inferOutputTensorSpecs(std::vector<TensorSpecs> inputs) {
		Shape newShape;
		
		// Remove dimensions along which the reduction is performed
		for (int i = 0; i < inputs[0].shape.size(); i++)
			if (!hasDim(i, m_Dim))
				newShape.push_back(inputs[0].shape[i]);

		return TensorSpecs{ inputs[0].dtype, newShape, inputs[0].device };
	}
}