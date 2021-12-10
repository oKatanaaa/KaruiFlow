#include "../headers/memory/DeviceCPU.h"
#include "../headers/memory/Exceptions.h"

namespace karuiflow {
	template<typename T>
	void add(void* _x, void* _y, void* _out, size_t n) {
		T* x = (T*)_x;
		T* y = (T*)_y;
		T* out = (T*)_out;

		for (int i = 0; i < n; i++)
			out[i] = x[i] + y[i];
	}

	void DeviceCPU::allocateMemory(void** ptr, size_t bytes) {
		*ptr = (void*)new char[bytes];
		if (*ptr == nullptr)
			throw MemoryAllocationError(bytes, "cpu");
	}

	void DeviceCPU::deallocateMemory(void* ptr) {
		delete ptr;
	}

	void DeviceCPU::copyDeviceToCpu(void* src, void* dst, size_t bytes) {
		memcpy(dst, src, bytes);
	}

	void DeviceCPU::copyCpuToDevice(void* src, void* dst, size_t bytes) {
		memcpy(dst, src, bytes);
	}

	void DeviceCPU::copyDeviceToDevice(void* src, void* dst, size_t bytes) {
		memcpy(dst, src, bytes);
	}

	void DeviceCPU::setZero(void* src, size_t bytes) {
		memset(src, 0, bytes);
	}

	std::string DeviceCPU::getDeviceName() {
		return "cpu";
	}

	int getDeviceId() {
		return 0;
	}

	std::function<void(void*, void*, void*, size_t)> DeviceCPU::getAdder(DType* dtype) {
		if (dtype->getName() == karuiflow::Float32().getName())
			return add<float>;
		else if (dtype->getName() == karuiflow::Int32().getName())
			return add<int>;
	}

}