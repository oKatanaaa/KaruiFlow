#include "../headers/Parameter.h"


namespace karuiflow {
	Parameter::Parameter(Tensor* tensor) {
		m_Data = tensor->m_Data->createSimilar();
		m_Data->copyFrom(tensor->m_Data);

		m_Specs = TensorSpecs{ tensor->m_Specs.dtype->copy(),  tensor->m_Specs.shape, tensor->m_Specs.device };
		m_RequiresGrad = true;
		incRefCount();
	}
}
