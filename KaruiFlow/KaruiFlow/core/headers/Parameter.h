#pragma once
#include "Tensor.h"


namespace karuiflow {
	/*
	* Represents parameters of a neural network. In fact it is just a mutable Tensor.
	*/
	class Parameter : public Tensor {

	public:
		Parameter(Storage* data, TensorSpecs specs, bool requiresGrad) :
			Tensor(data, specs, nullptr, std::vector<Tensor*>(), requiresGrad) {
			incRefCount();
		};
		Parameter(Tensor* tensor);

		void assign(Tensor* tensor) { m_Data->copyFrom(tensor->m_Data); };
		void assignAdd(Tensor* tensor) { m_Data->assignAdd(tensor->m_Data);	};
	};
		
}
