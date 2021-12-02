#pragma once

#include "Device.h"


namespace karuiflow {
	class DeviceCPU : public Device {
	public:
		DeviceCPU() {};

		void allocateMemory(void** ptr, size_t bytes);
		void deallocateMemory(void* ptr);
		void copyDeviceToCpu(void* src, void* dst, size_t bytes);
		void copyCpuToDevice(void* src, void* dst, size_t bytes);
		void copyDeviceToDevice(void* src, void* dst, size_t bytes);
		void setZero(void* src, size_t bytes);

		std::function<void(void*, void*, void*, size_t)> getAdder(DType* dtype);

		std::string getDeviceName();
	};
}