#pragma once
#include<string>

namespace karuiflow {
	class DType
	{
	protected:
		// Number of bytes occupied by this data type.
		size_t m_Size;
		std::string m_Name;

	protected:
		DType() { };

	public:
		size_t getSizeBytes() { return m_Size; }
		std::string getName() { return m_Name; }
	};


	class Float32 : public DType {
	public:
		Float32()
		{
			m_Size = sizeof(float);
			m_Name = "Float32";
		};
	};

	class Int32 : public DType {
	public:
		Int32()
		{
			m_Size = sizeof(int);
			m_Name = "Int32";
		};
	};
 }

