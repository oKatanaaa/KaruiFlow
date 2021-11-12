#pragma once
#include <string>
#include <functional>

#include "DType.h"

namespace karuiflow {

	class Device {
		/*
		* To make the storage independent of the specifics of memory allocation on different devices,
		* we use this instance for allocation on the given device.
		*/

	public:
		virtual void allocateMemory(void **ptr, size_t bytes) = 0;
		virtual void deallocateMemory(void** ptr) = 0;
		virtual void copyDeviceToCpu(void* src, void* dst, size_t bytes) = 0;
		virtual void copyCpuToDevice(void* src, void* dst, size_t bytes) = 0;
		virtual void copyDeviceToDevice(void* src, void* dst, size_t bytes) = 0;

		/*
		* Returns a function with the following interface:
		* void adder(void* x, void* y, void* out, size_t n)
		* Result is formed as: out[i] = x[i] + y[i]
		* This function can be used to add together two arrays that lie on the same device
		* and have the same number of elements.
		* 
		* In practice it will be used for gradient aggregation as well as for modification
		* of stateful variables in a graph (NN weights).
		* 
		* @param[in] dtype
		* Device will return an appropriate adder function depending on the data type (float, int, etc.)
		* 
		* @return
		* An approproate adder for the specified data type. Conversion of void* to the appropriate type
		* will be done inside of this function.
		*/
		virtual std::function<void(void*, void*, void*, size_t)> getAdder(DType dtype) = 0;

	public:
		virtual std::string getDeviceName() = 0;
		bool equalTo(Device* other) { return getDeviceName() == other->getDeviceName(); }
	};
}