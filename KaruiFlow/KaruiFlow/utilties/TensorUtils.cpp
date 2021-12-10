#pragma once
#include "TensorUtils.h"


namespace karuiflow {
	Tensor* toTensor(void* data, DType* dtype, Shape shape, bool requiresGrad) {
		Device* device = new DeviceCPU();
		Storage* storage = new Storage(dtype, shape, device);
		storage->copyData((void*)data);

		TensorSpecs specs = { dtype, shape, device };

		Tensor* tensor = new Tensor(storage, specs, requiresGrad);
	}

	Tensor* toTensor(float* data, Shape shape, bool requiresGrad) {
		return toTensor((void*)data, new Float32(), shape, requiresGrad);
	}


	Tensor* toTensor(int* data, Shape shape, bool requiresGrad) {
		return toTensor((void*)data, new Int32(), shape, requiresGrad);
	}
}
