#pragma once
#include "../../core/headers/Op.h"


namespace karuiflow {
	/*
	* Moves Tensor to the specified device.
	* Python code examples: 
	* - tensor.to("cuda:0") - moves tensor to CUDA device with id 0.
	* - tensor.to("cpu") - moves tensor to CPU side.
	*/
	class To : public Op {
	public:
		To(Device* device);

		std::string getOpName() override;

	protected:
		Kernel* instantiateKernel(std::vector<TensorSpecs> inputs) override;
		TensorSpecs inferOutputTensorSpecs(std::vector<TensorSpecs> inputs) override;

	private:
		Device* m_Device;
	};
}