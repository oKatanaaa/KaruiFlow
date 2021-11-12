#pragma once
#include "../Exception.h"
#include <string>

namespace karuiflow {
	class MemoryError : public Exception {};

	class MemoryAllocationError : public MemoryError {
	public:
		MemoryAllocationError(uint32_t bytes, std::string deviceName) {
			m_Message = "Failed to allocate" + std::to_string(bytes) +
				"on device=" + deviceName;
		}

		MemoryAllocationError(uint32_t bytes, const char* deviceName) {
			m_Message = std::string("Failed to allocate") + std::to_string(bytes) +
				"on device=" + std::string(deviceName);
		}
	};

	class MemoryCopyError : public MemoryError {};

}
