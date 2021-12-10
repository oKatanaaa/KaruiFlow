#include "../headers/memory/Device.h"


namespace karuiflow {
	std::string Device::getUniqueIdentifier() {
		return getDeviceName() + std::to_string(getDeviceId());
	}

	bool Device::equalTo(Device* other) {
		return getUniqueIdentifier() == other->getUniqueIdentifier();
	}
}