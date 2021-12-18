#include "SigmoidCPUKernels.h"


namespace karuiflow {
	SigmoidNumpyKernel::SigmoidNumpyKernel() {
		m_Obj = callPyGetSigmoid();
	}
}