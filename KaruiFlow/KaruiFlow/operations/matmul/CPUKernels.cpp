#include "CPUKernels.h"


namespace karuiflow {
	MatMulNumpyKernel::MatMulNumpyKernel() {
		m_Obj = callPyGetMatMul();
	}
}