#include "LogCPUKernels.h"


namespace karuiflow {
	LogNumpyKernel::LogNumpyKernel() {
		m_Obj = callPyGetLog();
	}
}