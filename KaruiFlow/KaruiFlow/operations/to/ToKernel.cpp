#include "ToKernel.h"


namespace karuiflow {
	void ToKernel::forward(std::vector<Storage*> inputs, Storage* output) {
		output->copyFrom(inputs[0]);
	}

	void ToKernel::backward(std::vector<Storage*> inputs, std::vector<bool> requiresGrad,
		Storage* outerGradient, std::vector<Storage*> outputGradients) {
		outputGradients[0]->copyFrom(outerGradient);
	}
}
