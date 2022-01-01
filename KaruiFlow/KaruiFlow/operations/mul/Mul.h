#pragma once

#include "../../core/headers/Op.h"

namespace karuiflow {

	class Log : public Op {
	public:
		Log() = default;

		std::string getOpName() override;

	protected:
		Kernel* instantiateKernel(std::vector<TensorSpecs> inputs) override;
		TensorSpecs inferOutputTensorSpecs(std::vector<TensorSpecs> inputs) override;
	};

}
