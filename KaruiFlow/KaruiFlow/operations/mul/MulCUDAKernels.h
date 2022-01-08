#pragma once
#include "../../core/headers/Kernel.h"
#include <cutensor.h>


namespace karuiflow {

	// Dispatches calls to Python side and uses Numpy to perform computation
	class MulCudaKernel : public Kernel {
	public:
		MulCudaKernel(DType* dtype);

		void forward(std::vector<Storage*> inputs, Storage* output);
		void backward(std::vector<Storage*> inputs, std::vector<bool> requiresGrad,
			Storage* outerGradient, std::vector<Storage*> outputGradients);
	private:

		/*
		* Initializes a tensor descriptor for the given storage.
		*
		* @param[out] desc
		* Reference to a descriptor to initialize.
		* @param[in] storage
		* The storage for which the descriptor is being initialized.
		* @param[in] notReduced
		* By default it must be set to `nullptr`. In this case modes and extents are computed as is.
		* If provided, the method will compute modes and extents relative to the tensor stored in `notReduced`.
		* It is assummed that the `storage` represents a reduced version (via summation, e.g. after applying this operation to `storage`)
		* of `notReduced`.
		* For terminology explanation (modes and extents) please refer to cuTensor documentation.
		*
		* @return modes
		* Modes computed for the tensor stored in `storage`.
		*/
		std::vector<int> initTensorDescriptor(cutensorTensorDescriptor_t& desc, Storage* storage, Storage* notReduced);
	private:
		std::vector<int> m_Dim;
		cudaDataType_t m_CuDtype;
		cutensorComputeType_t m_ComputeType;
	};
}

