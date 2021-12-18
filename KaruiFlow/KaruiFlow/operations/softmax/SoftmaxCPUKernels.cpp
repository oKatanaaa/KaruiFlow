#include "SoftmaxCPUKernels.h"


namespace karuiflow {
	SoftmaxNumpyKernel::SoftmaxNumpyKernel() {
		m_Obj = callPyGetSoftmax();
	}
}