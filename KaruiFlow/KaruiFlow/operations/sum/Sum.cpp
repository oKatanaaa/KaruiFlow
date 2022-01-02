#include <spdlog/spdlog.h>

#include "../../core/headers/LoggingUtils.h"
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
		// No axes specified, full tensor reduction
		if (m_Dim.size() == 0) {
			spdlog::debug("SumKernel // No axes specified, full tensor reduction.");
			return TensorSpecs{ inputs[0].dtype->copy(), Shape(), inputs[0].device };
		}

		Shape newShape;
		
		// Remove dimensions along which the reduction is performed
		for (int i = 0; i < inputs[0].shape.size(); i++)
			if (!hasDim(i, m_Dim))
				newShape.push_back(inputs[0].shape[i]);
		spdlog::debug("SumKernel // Specified axes:" + shapeToString(m_Dim));
		spdlog::debug("SumKernel // Output tensor shape:" + shapeToString(newShape));
		return TensorSpecs{ inputs[0].dtype->copy(), newShape, inputs[0].device };
	}
}
