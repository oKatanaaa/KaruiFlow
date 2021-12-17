IF CIMPORTS == 1:
    from .python_kernel_declaration cimport PyPythonKernel

cimport numpy as cnp

cdef class MatMulKernel(PyPythonKernel):
    cpdef void forward(self, list inputs, cnp.ndarray output)

    cpdef void backward(self, list inputs, list requiresGrad,
                      cnp.ndarray outerGradient, list outputGradients)