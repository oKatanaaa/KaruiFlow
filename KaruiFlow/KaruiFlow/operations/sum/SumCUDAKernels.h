#pragma once
#include "../../core/headers/Kernel.h"


namespace karuiflow {

	// Dispatches calls to Python side and uses Numpy to perform computation
	class SumCudaKernel : public Kernel {
	public:
		SumCudaKernel();
		SumCudaKernel(std::vector<int> dim);

		void forward(std::vector<Storage*> inputs, Storage* output);
		void backward(std::vector<Storage*> inputs, std::vector<bool> requiresGrad,
			Storage* outerGradient, std::vector<Storage*> outputGradients);

	private:
		std::vector<int> m_Dim;
		// Each letter is used to denote a single dimension of a tensor.
		const char* m_ModeAlphabet = "abcdefg";
	};
}

