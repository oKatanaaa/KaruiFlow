#pragma once

#include "../../core/headers/Op.h"

namespace karuiflow {

	class Mul : public Op {
	public:
		Mul() = default;

		std::string getOpName() override;

	protected:
		Kernel* instantiateKernel(std::vector<TensorSpecs> inputs) override;
		TensorSpecs inferOutputTensorSpecs(std::vector<TensorSpecs> inputs) override;
	};

}
