#include "ReluCPUKernels.h"


namespace karuiflow {
	ReluNumpyKernel::ReluNumpyKernel() {
		m_Obj = callPyGetRelu();
	}
}