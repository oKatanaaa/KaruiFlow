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
}
