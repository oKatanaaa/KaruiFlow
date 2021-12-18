#pragma once
#include "../../core/headers/PythonKernel.h"


namespace karuiflow {
	extern "C" PyObject * callPyGetSum(std::vector<int> dim);

	// Dispatches calls to Python side and uses Numpy to perform computation
	class SumNumpyKernel : public PythonKernel {
	public:
		SumNumpyKernel();
		SumNumpyKernel(std::vector<int> dim);

	private:
		std::vector<int> m_Dim;
	};
}

