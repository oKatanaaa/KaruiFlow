#pragma once
#include "../../core/headers/PythonKernel.h"


namespace karuiflow {
	extern "C" PyObject* callPyGetMatMul();

	// Dispatches calls to Python side and uses Numpy to perform computation
	class MatMulNumpyKernel : public PythonKernel {
	public:
		MatMulNumpyKernel();
	};
}

