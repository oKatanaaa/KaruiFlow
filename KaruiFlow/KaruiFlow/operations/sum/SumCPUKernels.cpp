#include "SumCPUKernels.h"


namespace karuiflow {
	SumNumpyKernel::SumNumpyKernel(std::vector<int> dim) : m_Dim(dim) {
		m_Obj = callPyGetSum(m_Dim);
	};


	SumNumpyKernel::SumNumpyKernel() {
		m_Obj = callPyGetSum(m_Dim);
	}
}