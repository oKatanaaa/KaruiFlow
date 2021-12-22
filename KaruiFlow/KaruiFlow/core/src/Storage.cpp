#include <functional>

#include "../headers/memory/Storage.h"
#include "../headers/memory/Exceptions.h"
#include <spdlog/spdlog.h>

namespace karuiflow {

	size_t Storage::getSize() {
		size_t size = 1;
		for (auto dim : m_Shape) {
			size *= dim;
		}
		return size;
	}

	size_t Storage::getSizeBytes() {
		return getSize() * m_Dtype->getSizeBytes();
	}

	void Storage::initialize() {
		size_t nbytes = getSizeBytes();
		m_Device->allocateMemory(&m_Data, nbytes);
	}

	void Storage::destroy() {
		m_Device->deallocateMemory(m_Data);
	}

	void Storage::copyFrom(Storage* other) {
		size_t bytes = getSizeBytes();
		if (bytes != other->getSizeBytes()) {
			std::string msg = "Storages are not of the same size.";
			msg += std::string("Expected ") + std::to_string(getSizeBytes()) + std::string("bytes, ");
			msg += std::string("but received ") + std::to_string(other->getSizeBytes());
			throw std::runtime_error(Exception(msg).what());
		}

		std::string thisDeviceName = m_Device->getDeviceName();
		std::string otherDeviceName = other->m_Device->getDeviceName();
		if (thisDeviceName == otherDeviceName)
			m_Device->copyDeviceToDevice(other->m_Data, m_Data, bytes);
		else if (thisDeviceName == "cpu" && otherDeviceName != "cpu")
			other->m_Device->copyDeviceToCpu(other->m_Data, m_Data, bytes);
		else if (thisDeviceName != "cpu" && otherDeviceName == "cpu")
			m_Device->copyCpuToDevice(other->m_Data, m_Data, bytes);
		else if (thisDeviceName != "cpu" && otherDeviceName != "cpu") {
			// In this case both devices are different and are not CPU.
			// For example, two different GPUs.
			// We need to transfer data from the other storage to CPU side
			// and then from CPU size transfer the data to this storage.
			char* buff = new char[bytes];
			other->m_Device->copyDeviceToCpu(other->m_Data, buff, bytes);
			m_Device->copyCpuToDevice(buff, m_Data, bytes);
			delete[] buff;
		}
	}

	void Storage::copyFrom(void* data) {
		m_Device->copyCpuToDevice(data, m_Data, getSizeBytes());
	}

	void Storage::copyTo(void* data) {
		m_Device->copyDeviceToCpu(m_Data, data, getSizeBytes());
	}

	void Storage::setZeros() {
		m_Device->setZeros(m_Data, getSizeBytes());
	}

	void Storage::assignAdd(Storage* other) {
		if (m_Device->getDeviceName() != other->m_Device->getDeviceName()) {
			std::string msg = "Cannot do assignAdd with storage because devices are different.";
			msg += "Expected device=" + m_Device->getDeviceName() + ", but received ";
			msg += other->m_Device->getDeviceName();
			throw std::runtime_error(Exception(msg).what());
		}

		spdlog::debug("Adding two storages.");
		// Full interface is: void adder(void* x, void* y, void* out, int n)
		std::function<void(void*, void*, void*, size_t)> adder = m_Device->getAdder(m_Dtype);
		
		adder(m_Data, other->m_Data, m_Data, getSize());
		spdlog::debug("Addition has been finished.");
	}

	Storage* Storage::createSimilar() {
		return new Storage(m_Dtype->copy(), m_Shape, m_Device);
	}

	Storage::~Storage() {
		destroy();
		delete m_Dtype;
	}
}
