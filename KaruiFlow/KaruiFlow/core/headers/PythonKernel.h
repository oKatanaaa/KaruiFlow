#pragma once
#define PY_SSIZE_T_CLEAN

#include "memory/Memory.h"
#include "Kernel.h"


struct _object;
typedef _object PyObject;


/*
* Both functions will be declared in Cython as Public API in order to be accessible
* from C++ code.
* The functions are used to call the Cython\Python versions of the corresponding methods.
 */
extern "C" void callPyForward(PyObject * obj, std::vector<karuiflow::Storage*> inputs, karuiflow::Storage * output);
extern "C" void callPyBackward(PyObject * obj, std::vector<karuiflow::Storage*> inputs, std::vector<bool> requiresGrad,
	karuiflow::Storage* outerGradient, std::vector<karuiflow::Storage*> outputGradients);


namespace karuiflow {

	class PythonKernel : public Kernel {
	public:
		PythonKernel();
		PythonKernel(PyObject* obj);

		void forward(std::vector<Storage*> inputs, Storage* output);
		void backward(std::vector<Storage*> inputs, std::vector<bool> requiresGrad,
			Storage* outerGradient, std::vector<Storage*> outputGradients);

	private:
		PyObject* m_Obj;
	};

}