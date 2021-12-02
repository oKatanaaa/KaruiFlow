#pragma once

#include "Op.h"
#include "PythonKernel.h"


struct _object;
typedef _object PyObject;


/*
* Both functions will be declared in Cython as Public API in order to be accessible
* from C++ code.
* The functions are used to call the Cython\Python versions of the corresponding methods.
 */
extern karuiflow::PythonKernel* callPyInstantiateKernel(PyObject* obj, std::vector<karuiflow::TensorSpecs> inputs);
extern karuiflow::TensorSpecs callPyInferOutputTensorSpecs(PyObject* obj, std::vector<karuiflow::TensorSpecs> inputs);


namespace karuiflow {
	class PythonOp : protected Op {
	public:
		PythonOp();
		PythonOp(PyObject* obj) : m_Obj(obj) {};

	protected:
		Kernel* instantiateKernel(std::vector<TensorSpecs> inputs);
		TensorSpecs inferOutputTensorSpecs(std::vector<TensorSpecs> inputs);

	private:
		PyObject* m_Obj;
	};
}
