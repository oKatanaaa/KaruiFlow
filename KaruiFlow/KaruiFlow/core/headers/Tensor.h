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

	class Tensor {
		friend class Parameter;

	public:
		Tensor() = default;
		Tensor(
			Storage* data, TensorSpecs specs, Kernel* parentOp,
			std::vector<Tensor*> inputTensors, bool requiresGrad);

		// Used in cases when the tensor is created by user, not by an operation
		Tensor(Storage* data, TensorSpecs specs, bool requiresGrad) : 
			m_Data(data), m_Specs(specs), m_ParentOp(nullptr), 
			m_InputTensors(std::vector<Tensor*>()), m_RequiresGrad(requiresGrad) {
			incRefCount();
		};

		~Tensor();

	public:
		TensorSpecs getTensorSpecs() { return m_Specs; }
		Storage* getDataStorage() { return m_Data; }
		Storage* getGradientStorage() { return m_Gradient; }
		Tensor* getGradient();
		Tensor* clone();
		void setRequiresGrad(bool requiresGrad) { m_RequiresGrad = requiresGrad; }
		bool requiresGrad() { return m_RequiresGrad; }
		bool isLeaf() { return m_ParentOp == nullptr; }

		/*
		* Invokes backpropagation for this tensor and all the
		* previous tensors in the computational graph that require gradient.
		*/
		void backward(Storage* outerGrad);
		void zeroGradient();
		void copyGradientTo(void* data);
		void copyTo(void* data);

		/* 
		* Objects is Python are being deleted all the time and in the case of Tensor
		* we cannot tie its lifetime to the lifetime of its Python wrapper. The reason
		* is that several Tensors may construct a graph and if some nodes are deleted from
		* that graph, it will cause segmentation fault during backward propagation.
		* Therefore, lifetime of a C++ Tensor is controlled separatly via reference
		* counting mechanism. Once reference count of a Tensor reaches zero, the
		* Tensor destroys itself.
		*/
		void incRefCount();
		void decRefCount();
		
	protected:
		void initGradient();

	protected:
		Storage* m_Data;
		Storage* m_Gradient = nullptr;
		Kernel* m_ParentOp = nullptr;
		TensorSpecs m_Specs;
		bool m_RequiresGrad = false;
		bool m_GradInitialized = false;
		std::vector<Tensor*> m_InputTensors;
		int m_ReferenceCounter = 0;
	};
}
