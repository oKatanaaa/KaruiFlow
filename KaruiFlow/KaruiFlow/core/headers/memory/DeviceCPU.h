#pragma once

#include "Device.h"


namespace karuiflow {
	class DeviceCPU : public Device {
	public:
		DeviceCPU() {};

		void allocateMemory(void** ptr, size_t bytes) override;
		void deallocateMemory(void* ptr) override;
		void copyDeviceToCpu(void* src, void* dst, size_t bytes) override;
		void copyCpuToDevice(void* src, void* dst, size_t bytes) override;
		void copyDeviceToDevice(void* src, void* dst, size_t bytes) override;
		void setZero(void* src, size_t bytes) override;

		std::function<void(void*, void*, void*, size_t)> getAdder(DType* dtype) override;

		std::string getDeviceName() override;
		int getDeviceId() override;
	};
}