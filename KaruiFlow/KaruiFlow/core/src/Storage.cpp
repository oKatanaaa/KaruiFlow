#include <functional>
#include <spdlog/spdlog.h>
#include <memory>

#include "../headers/memory/Storage.h"
#include "../headers/memory/Exceptions.h"
#include "../headers/memory/DeviceCPU.h"
#include "../headers/LoggingUtils.h"


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
		spdlog::debug("setZero in storage with shape " + shapeToString(m_Shape));
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

	void Storage::assignAdd(void* data, DType* dtype) {
		if (!dtype->equalTo(m_Dtype)) {
			std::string msg = "Cannot assignAdd data because of DType incompatibility: ";
			msg += "expected " + m_Dtype->getName() + ", ";
			msg += "but received " + dtype->getName() + ".";
			throw std::runtime_error(msg);
		}

		/*
		* Steps are as follows:
		* 1. Move storage's data to the CPU side.
		* 2. Perform addition.
		* 3. Move the result back to the device side.
		*/

		// Move data to a temporary buffer
		void* buff = (void*)new char[getSizeBytes()];
		m_Device->copyDeviceToCpu(m_Data, buff, getSizeBytes());

		// Perform addition
		DeviceCPU tempDevice;
		// Full interface is: void adder(void* x, void* y, void* out, int n)
		std::function<void(void*, void*, void*, size_t)> adder = tempDevice.getAdder(m_Dtype);
		adder(data, buff, buff, getSize());

		// Move data back to 
		m_Device->copyCpuToDevice(buff, m_Data, getSizeBytes());

		delete[] buff;
	}

	Storage* Storage::createSimilar() {
		return new Storage(m_Dtype->copy(), m_Shape, m_Device);
	}

	Storage::~Storage() {
		spdlog::debug("Deallocating storage with shape " + shapeToString(m_Shape));
		destroy();
		spdlog::debug("Deallocated memory buffer.");
		delete m_Dtype;
		spdlog::debug("Deallocated DType instance.");

		spdlog::debug("Deallocated storage.");
	}
}
