#pragma once
#include <vector>
#include "DType.h"
#include "Device.h"

namespace karuiflow {

	/*
	* A high-level representation of a memory chunk on a given device.
	* Can store only numerical types.
	*/
	class Storage {
	public:
		Storage(DType* dtype, std::vector<int> shape, Device* device) :
			m_Dtype(dtype), m_Shape(shape), m_Device(device) {
			initialize();
		};
		~Storage();

	public:
		static Storage* createSimilar(Storage* other);

	private:
		void initialize();

	public:
		std::vector<int> getShape() { return m_Shape; }
		size_t getSize();
		/*
		* Returns number of bytes used by this storage.
		*/
		size_t getSizeBytes();

		DType* getDtype() { return m_Dtype; }
		Device* getDevice() { return m_Device; }
		void* getData() { return m_Data; }

	public:
		/*
		* Deallocates all the memory from this storage.
		*/
		void destroy();

		/*
		* Copies data from the provided storage into the current Storage.
		*
		* @param other
		* Storage which to copy the data from.
		*/
		void copyData(Storage* other);
		void assignAdd(Storage* other);

		void setZero();
	
	private:
		DType* m_Dtype;
		Device* m_Device;
		std::vector<int> m_Shape;
		void* m_Data;
	};
}