#include "AddCPUKernels.h"


namespace karuiflow {
	AddNumpyKernel::AddNumpyKernel() {
		m_Obj = callPyGetAdd();
	}
}
