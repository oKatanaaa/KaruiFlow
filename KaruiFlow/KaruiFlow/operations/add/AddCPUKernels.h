#pragma once
#include "../../core/headers/PythonKernel.h"


namespace karuiflow {
	extern "C" PyObject * callPyGetAdd();

	// Dispatches calls to Python side and uses Numpy to perform computation
	class AddNumpyKernel : public PythonKernel {
	public:
		AddNumpyKernel();
	};
}

