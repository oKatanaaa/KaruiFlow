#pragma once
#include <vector>

#include "memory/Memory.h"


namespace karuiflow {

	/*
	* Does the actual mathematics.
	* All kernels must have forward propagation method (computes the mathematical function
	* the kernel represents) and backward propagation method (computes gradient of the function
	* with respect to its arguments).
	* 
	* Kernels are specialized by data types, data shape and devices.
	* Kernels assume that all the input Storages have appropriate meta data (dtype, shape and device).
	*/
	class Kernel {
	public:
		/*
		* Computes the math function.
		* 
		* @param inputs
		* Arguments of the function (the actual data to process). Order matters.
		* @param output
		* Where to store the result.
		*/
		virtual void forward(std::vector<Storage*> inputs, Storage* output) = 0;

		/*
		* Computes gradient of the math function with respect to its arguments.
		* 
		* @param[in] inputs
		* Arguments of the function (the actual data to process). Order matters.
		* @param[in] requiresGrad
		* A vector of booleans. If ith element is False, that means that the ith input
		* does not require the gradient and it should not be computed to save memory and compute.
		* @param[in] outerGradient
		* Used for computing jacobian vector products.
		* @param[out] outputGradients
		* Where to store the computed gradients. Order must be the same as in `inputs`.
		* It might be the case that some of the pointers are NULL. That means that the corresponding
		* tensor does not require gradients.
		*/
		virtual void backward(std::vector<Storage*> inputs, std::vector<bool> requiresGrad, 
			Storage* outerGradient, std::vector<Storage*> outputGradients) = 0;
	};
}
