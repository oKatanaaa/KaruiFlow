#pragma once
#include "../core/headers/KaruiFlowCore.h"

namespace karuiflow {
	Tensor* toTensor(void* data, DType* dtype, Shape shape, bool requiresGrad);
	Tensor* toTensor(float* data, Shape shape, bool requiredGrad);
	Tensor* toTensor(int* data, Shape shape, bool requiredGrad);
}
