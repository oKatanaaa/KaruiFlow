#pragma once

#include "../../core/headers/Op.h"

namespace karuiflow {

	class Sum : public Op {
	public:
		Sum() = default;
		Sum(std::vector<int> dim) : m_Dim(dim) { };

		std::string getOpName() override;

	protected:
		Kernel* instantiateKernel(std::vector<TensorSpecs> inputs) override;
		TensorSpecs inferOutputTensorSpecs(std::vector<TensorSpecs> inputs) override;

	private:
		std::vector<int> m_Dim;
	};

}
