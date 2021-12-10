#pragma once
#include <vector>

#include "memory/Memory.h"
#include "Kernel.h"


namespace karuiflow {

	typedef std::vector<int> Shape;

	struct TensorSpecs {
		DType* dtype;
		Shape shape;
		Device* device;
	};

	std::string shapeToString(Shape& shape);

	class Tensor {
		friend class Parameter;

	public:
		Tensor(
			Storage* data, TensorSpecs specs, Kernel* parentOp,
			std::vector<Tensor*> inputTensors, bool requiresGrad) :
			m_Data(data), m_Specs(specs), m_ParentOp(parentOp),
			m_InputTensors(inputTensors), m_RequiresGrad(requiresGrad) {};

		// Used in cases when the tensor is created by user, not by an operation
		Tensor(Storage* data, TensorSpecs specs, bool requiresGrad) : 
			m_Data(data), m_Specs(specs), m_ParentOp(nullptr), 
			m_InputTensors(std::vector<Tensor*>()), m_RequiresGrad(requiresGrad) {};

	public:
		TensorSpecs getTensorSpecs() { return m_Specs; }
		Storage* getDataStorage() { return m_Data; }
		Storage* getGradientStorage() { return m_Gradient; }
		void setRequiresGrad(bool requiresGrad) { m_RequiresGrad = requiresGrad; }
		bool requiresGrad() { return m_RequiresGrad; }

	public:
		/*
		* Invokes backpropagation for this tensor and all the
		* previous tensors in the computational graph that require gradient.
		*/
		void backward(Storage* outerGrad);
		void zeroGradient();

	protected:
		Tensor() = delete;
		void initGradient();

	protected:
		Storage* m_Data;
		Storage* m_Gradient = nullptr;
		Kernel* m_ParentOp;
		const TensorSpecs m_Specs;
		bool m_RequiresGrad;
		bool m_GradInitialized = false;
		std::vector<Tensor*> m_InputTensors;
	};
}
