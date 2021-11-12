#pragma once
#include <vector>

#include "memory/Memory.h"
#include "Kernel.h"
#include "Tensor.h"


namespace karuiflow {

	/*
	* Represents a high-level concept of a mathematical operation.
	* This class is responsible for instantiating appropriate kernels depending
	* on the inputs' TensorSpecs as well as creation of new Tensors.
	*/
	class Op {
	public:
		Tensor* operator()(std::vector<Tensor*> inputs);
		virtual std::string getOpName() = 0;

	private:
		void assertDeviceSame(std::vector<TensorSpecs> inputs);
		bool isRequiresGrad(std::vector<Tensor*> inputs);

	protected:
		/*
		* Instatiates an appropriate kernel depending on the inputs' specs.
		* 
		* @param[in] inputs
		* A list of TensorSpecs. Used to determine the appropriate kernel.
		* @param[out] Kernel
		* Pointer to the instantiated kernel.
		*/
		virtual Kernel* instantiateKernel(std::vector<TensorSpecs> inputs) = 0;

		/*
		* Infers TensorSpecs for the output tensor.
		*
		* @param[in] inputs
		* A list of TensorSpecs. Used to infer TensorSpecs for the output tensor.
		* @param[out] TensorSpecs
		* The infered TensorSpecs.
		*/
		virtual TensorSpecs inferOutputTensorSpecs(std::vector<TensorSpecs> inputs) = 0;
	};

	
	class OpException : public Exception {
	public:
		OpException(std::string opName, std::string msg) {
			m_Message += "Exception in operation: " + opName + ". Message: " + msg;
		}
	};
}