#include <spdlog/spdlog.h>


namespace karuiflow {

	void setDebugLogLevel() {
		spdlog::set_level(spdlog::level::debug);
	}

	void setInfoLogLevel() {
		spdlog::set_level(spdlog::level::info);
	}

	void setErrLogLevel() {
		spdlog::set_level(spdlog::level::err);
	}

	void setWarnLogLevel() {
		spdlog::set_level(spdlog::level::warn);
	}

	std::string shapeToString(std::vector<int>& shape) {
		std::string str;
		str += "[";
		for (int i = 0; i < shape.size(); i++) {
			str += std::to_string(shape[i]);
			if (i + 1 < shape.size())
				str += ", ";
		}
		str += "]";
		return str;
	}
}
