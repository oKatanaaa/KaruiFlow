#pragma once
#include <string>
#include <iostream>


namespace karuiflow {

	class Exception {
	protected:
		std::string m_Message;

	public:
		Exception() {};
		Exception(std::string msg) : m_Message(msg) {}
		Exception(const char* msg) : m_Message(msg) {}

		std::string getMessage() { return m_Message; }
		std::string printMessage() { std::cout << m_Message << std::endl; }
	};


}