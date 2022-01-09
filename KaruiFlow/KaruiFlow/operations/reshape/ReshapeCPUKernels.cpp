#include "ReshapeCPUKernels.h"


namespace karuiflow {
	void ReshapeKernel::forward(std::vector<Storage*> inputs, Storage* output) {
		output->copyFrom(inputs[0]);
	}
	void ReshapeKernel::backward(std::vector<Storage*> inputs, std::vector<bool> requiresGrad,
		Storage* outerGradient, std::vector<Storage*> outputGradients) {
		if (requiresGrad[0])
			outputGradients[0]->copyFrom(outerGradient);
	}
}
