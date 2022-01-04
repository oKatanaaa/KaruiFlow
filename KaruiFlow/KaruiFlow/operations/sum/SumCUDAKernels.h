#pragma once
#include "../../core/headers/Kernel.h"


namespace karuiflow {

	// Dispatches calls to Python side and uses Numpy to perform computation
	class SumCudaKernel : public Kernel {
	public:
		SumCudaKernel(std::vector<int> dim) : m_Dim(dim) {};

		void forward(std::vector<Storage*> inputs, Storage* output);
		void backward(std::vector<Storage*> inputs, std::vector<bool> requiresGrad,
			Storage* outerGradient, std::vector<Storage*> outputGradients);

	private:
		std::vector<int> m_Dim;
	};
}

