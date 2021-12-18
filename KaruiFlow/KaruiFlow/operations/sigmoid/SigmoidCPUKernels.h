#pragma once
#include "../../core/headers/PythonKernel.h"


namespace karuiflow {
	extern "C" PyObject * callPyGetSigmoid();

	// Dispatches calls to Python side and uses Numpy to perform computation
	class SigmoidNumpyKernel : public PythonKernel {
	public:
		SigmoidNumpyKernel();
	};
}

