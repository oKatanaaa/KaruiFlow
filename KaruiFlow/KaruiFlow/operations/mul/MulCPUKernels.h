#pragma once
#include "../../core/headers/PythonKernel.h"


namespace karuiflow {
	extern "C" PyObject * callPyGetMul();

	// Dispatches calls to Python side and uses Numpy to perform computation
	class MulNumpyKernel : public PythonKernel {
	public:
		MulNumpyKernel();
	};
}

