#include "Reshape.h"
#include "ReshapeCPUKernels.h"
#include "../../core/headers/LoggingUtils.h"


namespace karuiflow {
	Reshape::Reshape(std::vector<int> newShape) {
		m_Size = 1;
		for (int dim : newShape) {
			if (dim <= 0)
				throw std::runtime_error("Encountered non positive dimension in new shape. Received shape:" + shapeToString(newShape));
			m_Size *= dim;
		}

		m_NewShape = newShape;
	}

	std::string Reshape::getOpName() {
		return "Reshape";
	}

	Kernel* Reshape::instantiateKernel(std::vector<TensorSpecs> inputs) {
		return new ReshapeKernel();
	}

	TensorSpecs Reshape::inferOutputTensorSpecs(std::vector<TensorSpecs> inputs) {
		auto inputShape = inputs[0].shape;
		size_t inputSize = 1;
		for (int dim : inputShape)
			inputSize *= dim;

		if (inputSize != m_Size)
			throw std::runtime_error("Cannot reshape tensor with " + std::to_string(inputSize) +
				" elems into a tensor with " + std::to_string(m_Size) + " elems."
			);

		return TensorSpecs{ inputs[0].dtype->copy(), m_NewShape, inputs[0].device };
	}
}
