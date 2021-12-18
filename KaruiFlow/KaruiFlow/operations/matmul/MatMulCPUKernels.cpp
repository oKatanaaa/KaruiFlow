#include "MatMulCPUKernels.h"


namespace karuiflow {
	MatMulNumpyKernel::MatMulNumpyKernel() {
		m_Obj = callPyGetMatMul();
	}
}