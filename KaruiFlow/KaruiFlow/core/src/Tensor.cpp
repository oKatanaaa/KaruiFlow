#pragma once

#include "../headers/Tensor.h"
#include <spdlog/spdlog.h>
#define KF_ERROR(exc) (std::runtime_error(exc.what()))

namespace karuiflow {

	std::string shapeToString(Shape& shape) {
		std::string str;
		str += "[";
		for (int i = 0; i < shape.size(); i++) {
			str += std::to_string(shape[i]);
			if (i + 1 < shape.size())
				str += ", ";
		}
		str += "]";
		return str;
	}

	void Tensor::initGradient() {
		m_Gradient = new Storage(m_Specs.dtype, m_Specs.shape, m_Specs.device);
		m_GradInitialized = true;
	}

	void Tensor::backward(Storage* outerGrad) {
		spdlog::debug("Executing backward call in a Tensor.");

		if (!m_RequiresGrad) {
			spdlog::debug("The tensor does not require gradient. Returning...");
			return;
		}

		if (!m_GradInitialized) {
			spdlog::debug("Initializing gradient storage in the Tensor.");
			initGradient();
		}

		// Accumulate incoming gradient
		if (outerGrad == nullptr) {
			spdlog::error("The tensor requires gradient, but received null pointer.");
			throw KF_ERROR(Exception("The tensor requires gradient, but received null pointer."));
		}

		spdlog::debug("Accumulating incoming gradient.");
		m_Gradient->assignAdd(outerGrad);

		if (m_ParentOp == nullptr) {
			// This is a leaf of the graph
			spdlog::debug("Tensor has no parent operation. Returning...");
			return;
		}

		spdlog::debug("Gathering metadata...");
		std::vector<Storage*> inputData(m_InputTensors.size());
		std::vector<bool> requiresGrad(m_InputTensors.size());
		std::vector<Storage*> inputGradients(m_InputTensors.size());
		for (int i = 0; i < m_InputTensors.size(); i++) {
			spdlog::debug("\n---------");
			spdlog::debug("Tensor[" + std::to_string(i) + "]:");

			inputData[i] = m_InputTensors[i]->getDataStorage();
			requiresGrad[i] = m_InputTensors[i]->requiresGrad();

			spdlog::debug("requiresGrad=" + std::to_string(requiresGrad[i]));
			spdlog::debug("dataStorageAddress=" + std::to_string((long)inputData[i]) + "\n");

			Storage* gradient = nullptr;
			if (requiresGrad[i])
				gradient = inputData[i]->createSimilar();
			inputGradients[i] = gradient;
		}
		spdlog::debug("Finished gathering metadata.");

		spdlog::debug("Calling backward in kernel.");
		m_ParentOp->backward(inputData, requiresGrad, outerGrad, inputGradients);

		// The storage for the outer gradient must be destroyed to free the resources.
		spdlog::debug("Deleting outer gradient storage.");
		delete outerGrad;

		spdlog::debug("Sending gradients to parent tensors.");
		for (int i = 0; i < m_InputTensors.size(); i++)
			m_InputTensors[i]->backward(inputGradients[i]);

		spdlog::debug("Gradients were sent successfully. Returning...");
		return;
	}

	void Tensor::zeroGradient() {
		if (!m_GradInitialized)
			throw std::runtime_error("Gradient storage has not been initialized.");
		m_Gradient->setZeros();
	}

	void Tensor::copyGradientTo(void* data) {
		if (!m_GradInitialized)
			throw std::runtime_error("Gradient storage has not been initialized.");
		m_Gradient->copyTo(data);
	}

	void Tensor::copyFrom(void* data) {
		m_Data->copyFrom(data);
	}

	void Tensor::copyTo(void* data) {
		m_Data->copyTo(data);
	}
}
