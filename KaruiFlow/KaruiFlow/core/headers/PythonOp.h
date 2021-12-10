#pragma once

#include "Op.h"
#include "Kernel.h"


struct _object;
typedef _object PyObject;


/*
* Both functions will be declared in Cython as Public API in order to be accessible
* from C++ code.
* The functions are used to call the Cython\Python versions of the corresponding methods.
 */
extern "C" karuiflow::Kernel * callPyInstantiateKernel(PyObject * obj, std::vector<karuiflow::TensorSpecs> inputs);
extern "C" karuiflow::TensorSpecs callPyInferOutputTensorSpecs(PyObject * obj, std::vector<karuiflow::TensorSpecs> inputs);
extern "C" std::string callPyGetOpName(PyObject * obj);

namespace karuiflow {
	class PythonOp : public Op {
	public:
		PythonOp() = default;
		PythonOp(PyObject* obj) : m_Obj(obj) {};

	public:
		Kernel* instantiateKernel(std::vector<TensorSpecs> inputs);
		TensorSpecs inferOutputTensorSpecs(std::vector<TensorSpecs> inputs);
		std::string getOpName();

	private:
		PyObject* m_Obj;
	};
}
