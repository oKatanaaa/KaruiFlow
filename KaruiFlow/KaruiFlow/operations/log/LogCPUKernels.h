#pragma once
#include "../../core/headers/PythonKernel.h"


namespace karuiflow {
	extern "C" PyObject * callPyGetLog();

	// Dispatches calls to Python side and uses Numpy to perform computation
	class LogNumpyKernel : public PythonKernel {
	public:
		LogNumpyKernel();
	};
}

