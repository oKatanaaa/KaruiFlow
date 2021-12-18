#pragma once
#include "../../core/headers/PythonKernel.h"


namespace karuiflow {
	extern "C" PyObject * callPyGetRelu();

	// Dispatches calls to Python side and uses Numpy to perform computation
	class ReluNumpyKernel : public PythonKernel {
	public:
		ReluNumpyKernel();
	};
}

