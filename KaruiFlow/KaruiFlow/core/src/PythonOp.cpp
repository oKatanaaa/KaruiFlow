#include "../headers/PythonOp.h"


namespace karuiflow {
	Kernel* PythonOp::instantiateKernel(std::vector<TensorSpecs> inputs) {
		if (m_Obj == nullptr)
			throw std::runtime_error("No PyObject was set.");

		return (Kernel*)callPyInstantiateKernel(m_Obj, inputs);
	}

	TensorSpecs PythonOp::inferOutputTensorSpecs(std::vector<TensorSpecs> inputs) {
		if (m_Obj == nullptr)
			throw std::runtime_error("No PyObject was set.");

		return callPyInferOutputTensorSpecs(m_Obj, inputs);
	}
}
