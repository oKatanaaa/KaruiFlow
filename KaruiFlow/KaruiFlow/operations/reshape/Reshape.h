#pragma once
#include "../../core/headers/Op.h"

namespace karuiflow {

	class Reshape : public Op {
	public:
		Reshape(std::vector<int> newShape);

		std::string getOpName() override;

	protected:
		Kernel* instantiateKernel(std::vector<TensorSpecs> inputs) override;
		TensorSpecs inferOutputTensorSpecs(std::vector<TensorSpecs> inputs) override;

	private:
		std::vector<int> m_NewShape;
		size_t m_Size;
	};

}