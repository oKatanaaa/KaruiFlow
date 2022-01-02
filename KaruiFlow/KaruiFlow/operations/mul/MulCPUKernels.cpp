#include "MulCPUKernels.h"


namespace karuiflow {
	MulNumpyKernel::MulNumpyKernel() {
		m_Obj = callPyGetMul();
	}
}