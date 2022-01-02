#include <spdlog/spdlog.h>

#include "../headers/Op.h"
#include "../headers/LoggingUtils.h"


namespace karuiflow {
	Tensor* Op::operator()(std::vector<Tensor*> inputs) {
		spdlog::debug("Calling" + getOpName() + "operation.");
		std::vector<TensorSpecs> inputSpecs;
		for (auto t : inputs)
			inputSpecs.push_back(t->getTensorSpecs());
		// Gathering meta data
		assertDeviceSame(inputSpecs);
		bool requiredGrad = isRequiresGrad(inputs);
		TensorSpecs specs = inferOutputTensorSpecs(inputSpecs);
		Kernel* kernel = instantiateKernel(inputSpecs);

		// Invoking the kernel and constructing the output tensor
		std::vector<Storage*> inputData;
		for (auto t : inputs)
			inputData.push_back(t->getDataStorage());
		Storage* output = new Storage(specs.dtype, specs.shape, specs.device);
		spdlog::debug("Calling forward in kernel.");
		kernel->forward(inputData, output);

		// Construct new Tensor
		Tensor* outTensor = new Tensor(output, specs, kernel, inputs, requiredGrad);
		return outTensor;
	}

	void Op::assertDeviceSame(std::vector<TensorSpecs> inputs) {
		Device* device = inputs[0].device;
		for (auto spec : inputs)
			if (!device->equalTo(spec.device))
				throw std::runtime_error(OpException(getOpName(), "Inconsistent devices among input tensors.").what());
	}

	bool Op::isRequiresGrad(std::vector<Tensor*> inputs) {
		bool requiresGrad = false;
		for (auto tensor : inputs)
			requiresGrad |= tensor->requiresGrad();
		return requiresGrad;
	}


	OpException::OpException(std::string opName, std::string msg) {
		m_Message += "Exception in operation: " + opName + ". Message: " + msg;
	}

	InconsistentShapes::InconsistentShapes(std::string opName, std::vector<Shape> shapes)
		: OpException(opName, "") {
		m_Message += "Received tensors with inconsistent shapes: ";
		for (Shape shape : shapes)
			m_Message += shapeToString(shape) + " ";
	}

	UnsuppotedShapes::UnsuppotedShapes(std::string opName, std::vector<Shape> shapes)
		: OpException(opName, "") {
		m_Message += "Received tensors with unsupported shapes: ";
		for (Shape shape : shapes)
			m_Message += shapeToString(shape) + " ";
	}
}