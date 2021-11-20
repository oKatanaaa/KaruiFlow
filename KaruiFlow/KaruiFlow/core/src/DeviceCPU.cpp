#include "../headers/memory/DeviceCPU.h"
#include "../headers/memory/Exceptions.h"

	template<typename T>
	void add(void* _x, void* _y, void* _out, size_t n) {
		T* x = (T*)_x;
		T* y = (T*)_y;
		T* out = (T*)_out;

		for (int i = 0; i < n; i++)
			out[i] = x[i] + y[i];
	}

	void karuiflow::DeviceCPU::allocateMemory(void** ptr, size_t bytes) {
		*ptr = (void*)new char[bytes];
		if (*ptr == nullptr)
			throw MemoryAllocationError(bytes, "cpu");
	}

	void karuiflow::DeviceCPU::deallocateMemory(void* ptr) {
		delete ptr;
	}

	void karuiflow::DeviceCPU::copyDeviceToCpu(void* src, void* dst, size_t bytes) {
		memcpy(dst, src, bytes);
	}

	void karuiflow::DeviceCPU::copyCpuToDevice(void* src, void* dst, size_t bytes) {
		memcpy(dst, src, bytes);
	}

	void karuiflow::DeviceCPU::copyDeviceToDevice(void* src, void* dst, size_t bytes) {
		memcpy(dst, src, bytes);
	}

	std::string karuiflow::DeviceCPU::getDeviceName() {
		return "cpu";
	}

	std::function<void(void*, void*, void*, size_t)> karuiflow::DeviceCPU::getAdder(karuiflow::DType dtype) {
		if (dtype.getName() == karuiflow::Float32().getName())
			return add<float>;
		else if (dtype.getName() == karuiflow::Int32().getName())
			return add<int>;
	}

