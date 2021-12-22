#pragma once

#include <exception>
#include <spdlog/spdlog.h>
#include "../headers/PythonKernel.h"


//extern "C" int PyImport_ImportModule(const char*);

namespace karuiflow {
	PythonKernel::PythonKernel() : m_Obj(nullptr) {};

	PythonKernel::PythonKernel(PyObject* obj) : m_Obj(obj) {};

	void PythonKernel::forward(std::vector<Storage*> inputs, Storage* output) {
		if (m_Obj == nullptr)
			throw std::runtime_error("No PyObject was set.");


		callPyForward(m_Obj, inputs, output);
	}

	void PythonKernel::backward(std::vector<Storage*> inputs, std::vector<bool> requiresGrad,
		Storage* outerGradient, std::vector<Storage*> outputGradients) {
		if (m_Obj == nullptr)
			throw std::runtime_error("No PyObject was set.");

		spdlog::debug("Calling backward in Python kernel.");
		callPyBackward(m_Obj, inputs, requiresGrad, outerGradient, outputGradients);
		spdlog::debug("Call finished.");
	}
}
