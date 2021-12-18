#pragma once
#include "../../core/headers/PythonKernel.h"


namespace karuiflow {
	extern "C" PyObject * callPyGetSoftmax();

	// Dispatches calls to Python side and uses Numpy to perform computation
	class SoftmaxNumpyKernel : public PythonKernel {
	public:
		SoftmaxNumpyKernel();
	};
}

