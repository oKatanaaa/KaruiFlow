#pragma once

#include "../headers/Tensor.h"


namespace karuiflow {

	void Tensor::initGradient() {
		m_Gradient = new Storage(m_Specs.dtype, m_Specs.shape, m_Specs.device);
		m_GradInitialized = true;
	}

	void Tensor::backward(Storage* outerGrad) {
		if (!m_RequiresGrad)
			return;

		if (!m_GradInitialized)
			initGradient();

		// Accumulate incoming gradient
		if (outerGrad == nullptr)
			throw Exception("The tensor requires gradient, but received null pointer.");
		m_Gradient->assignAdd(outerGrad);

		if (m_ParentOp == nullptr)
			// This is a leaf of the graph
			return;

		std::vector<Storage*> inputData;
		std::vector<bool> requiresGrad;
		for (int i = 0; i < m_InputTensors.size(); i++) {
			inputData.push_back(m_InputTensors[i]->getDataStorage());
			requiresGrad.push_back(m_InputTensors[i]->requiresGrad());
		}

		std::vector<Storage*> inputGradients = m_ParentOp->backward(inputData, requiresGrad, m_Gradient);

		// The storage for the outer gradient must be destroyed to free the resources.
		delete outerGrad;

		for (int i = 0; i < m_InputTensors.size(); i++)
			m_InputTensors[i]->backward(inputGradients[i]);

	}
}