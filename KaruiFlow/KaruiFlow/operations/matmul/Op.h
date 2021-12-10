#pragma once

#include "../../core/headers/Op.h"

namespace karuiflow {

	class MatMul : public Op {

	protected:
		Kernel* instantiateKernel(std::vector<TensorSpecs> inputs) override;
		TensorSpecs inferOutputTensorSpecs(std::vector<TensorSpecs> inputs) override;
	};

}
