#pragma once
#include <spdlog/spdlog.h>
#include <spdlog/fmt/fmt.h>

#include "../headers/Tensor.h"
#include "../headers/LoggingUtils.h"


#define KF_ERROR(exc) (std::runtime_error(exc.what()))

namespace karuiflow {
	Tensor::Tensor(
		Storage* data, TensorSpecs specs, Kernel* parentOp,
		std::vector<Tensor*> inputTensors, bool requiresGrad) :
		m_Data(data), m_Specs(specs), m_ParentOp(parentOp),
		m_InputTensors(inputTensors), m_RequiresGrad(requiresGrad)
	{
		incRefCount();
		for (Tensor* tensor : inputTensors)
			tensor->incRefCount();
	};

	Tensor::~Tensor() {
		spdlog::debug("Deallocating tensor...");
		delete m_Data;
		spdlog::debug("Deallocated data.");
		if (m_GradInitialized) {
			delete m_Gradient;
			spdlog::debug("Deallocated gradient.");
		}
		if (m_ParentOp != nullptr) {
			delete m_ParentOp;
			spdlog::debug("Deallocated parent op.");
		}
		if (m_InputTensors.size() != 0)
			for (Tensor* tensor : m_InputTensors)
				tensor->decRefCount();
		spdlog::debug("Successfully deallocated tensor.");
	}

	void Tensor::initGradient() {
		m_Gradient = new Storage(m_Specs.dtype->copy(), m_Specs.shape, m_Specs.device);
		m_Gradient->setZeros();
		m_GradInitialized = true;
	}

	Tensor* Tensor::getGradient() {
		if (!m_GradInitialized)
			throw std::runtime_error("Gradient has not been initialized.");
		Storage* gradient = m_Gradient->createSimilar();
		gradient->copyFrom(m_Gradient);
		Tensor* tensor = new Tensor(gradient, m_Specs, false);
		return tensor;
	}

	Tensor* Tensor::clone() {
		Storage* dataCopy = m_Data->createSimilar();
		dataCopy->copyFrom(m_Data);
		TensorSpecs specs = TensorSpecs{ m_Specs.dtype->copy(), m_Specs.shape, m_Specs.device };
		return new Tensor(dataCopy, specs, false);
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
		std::vector<bool> requiresGradList(m_InputTensors.size());
		std::vector<Storage*> inputGradients(m_InputTensors.size());
		for (int i = 0; i < m_InputTensors.size(); i++) {
			spdlog::debug("\n---------");
			spdlog::debug("Tensor[" + std::to_string(i) + "]:");

			inputData[i] = m_InputTensors[i]->getDataStorage();
			requiresGradList[i] = m_InputTensors[i]->requiresGrad();

			spdlog::debug("requiresGradList=" + std::to_string(requiresGradList[i]));
			spdlog::debug("dataStorageAddress=" + std::to_string((long)inputData[i]) + "\n");

			Storage* gradient = nullptr;
			if (requiresGradList[i])
				gradient = inputData[i]->createSimilar();
			inputGradients[i] = gradient;
		}
		spdlog::debug("Finished gathering metadata.");

		spdlog::debug("Calling backward in kernel.");
		m_ParentOp->backward(inputData, requiresGradList, outerGrad, inputGradients);

		// The storage for the outer gradient must be destroyed to free the resources.
		//spdlog::debug("Deleting outer gradient storage.");
		//delete outerGrad;

		spdlog::debug("Sending gradients to parent tensors.");
		for (int i = 0; i < m_InputTensors.size(); i++)
			m_InputTensors[i]->backward(inputGradients[i]);

		for (int i = 0; i < m_InputTensors.size(); i++) {
			if (requiresGradList[i])
				delete inputGradients[i];
		}

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

	void Tensor::copyTo(void* data) {
		m_Data->copyTo(data);
	}

	void Tensor::to(Device* device) {
		if (!isLeaf())
			throw std::runtime_error("Called `to` on a non-leaf Tensor.");
		m_Data->to(device);
		if (m_GradInitialized)
			m_Gradient->to(device);
		// We cannot delete the device as it can be used by other objects.
		m_Specs.device = device;
	}

	void Tensor::incRefCount() {
		spdlog::debug("Increased reference count.");
		m_ReferenceCounter++;
	}

	void Tensor::decRefCount() {
		spdlog::debug("Decreased reference count.");
		m_ReferenceCounter--;

		if (m_ReferenceCounter <= 0) {
			spdlog::debug("Reference count met conditions for self-destruction.");
			delete this;
		}
	}
}
