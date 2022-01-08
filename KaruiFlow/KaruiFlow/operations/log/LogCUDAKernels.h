#pragma once
#include "../../core/headers/Kernel.h"


namespace karuiflow {

	class LogCudaKernel : public Kernel {
	public:
		LogCudaKernel() {};
		void forward(std::vector<Storage*> inputs, Storage* output) override;
		void backward(std::vector<Storage*> inputs, std::vector<bool> requiresGrad,
			Storage* outerGradient, std::vector<Storage*> outputGradients) override;

	private:
		int m_ThreadsPerBlock = 256;
	};
}
