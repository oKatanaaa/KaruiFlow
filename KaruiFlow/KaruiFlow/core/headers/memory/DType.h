#pragma once
#include <string>

namespace karuiflow {
	class DType
	{
	protected:
		// Number of bytes occupied by this data type.
		size_t m_Size;
		std::string m_Name;

	public:
		DType() : m_Size(0), m_Name("none") { };
		DType(size_t size, std::string name) : m_Size(size), m_Name(name) { };
		
	public:
		size_t getSizeBytes() { return m_Size; }
		std::string getName() { return m_Name; }
		DType* copy() { return new DType(m_Size, m_Name); }
		bool equalTo(DType* other) { return getName() == other->getName(); }
	};


	class Float32 : public DType {
	public:
		Float32()
		{
			m_Size = sizeof(float);
			m_Name = "float32";
		};
	};

	class Int32 : public DType {
	public:
		Int32()
		{
			m_Size = sizeof(int);
			m_Name = "int32";
		};
	};
 }

