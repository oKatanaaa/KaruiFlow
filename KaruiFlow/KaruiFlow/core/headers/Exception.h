#pragma once
#include <string>
#include <iostream>


namespace karuiflow {

	class Exception {
	protected:
		std::string m_Message;

	public:
		Exception() = default;
		Exception(std::string msg) : m_Message(msg) {}
		Exception(const char* msg) : m_Message(msg) {}

		std::string what() const { return m_Message; }
		void printMessage() { std::cout << m_Message << std::endl; }
	};


}