#include "../headers/memory/Device.h"
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

	class CPUDevice : public Device {
	public:
		void allocateMemory(void** ptr, int bytes) {
			*ptr = (void*)new char[bytes];
			if (*ptr == nullptr)
				throw MemoryAllocationError(bytes, "cpu");
		}

		void deallocateMemory(void** ptr) {
			delete[] *ptr;
		}

		void copyDeviceToCpu(void* src, void* dst, size_t bytes) {
			memcpy(dst, src, bytes);
		}

		void copyCpuToDevice(void* src, void* dst, size_t bytes) {
			memcpy(dst, src, bytes);
		}

		void copyDeviceToDevice(void* src, void* dst, size_t bytes) {
			memcpy(dst, src, bytes);
		}

		std::string getDeviceName() {
			return "cpu";
		}

		std::function<void(void*, void*, void*, size_t)> getAdder(DType dtype) {
			if (dtype.getName() == Float32().getName())
				return add<float>;
			else if (dtype.getName() == Int32().getName())
				return add<int>;
		}

	};

	
}