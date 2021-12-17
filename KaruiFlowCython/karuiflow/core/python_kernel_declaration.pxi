# distutils: language = c++

cimport numpy as cnp


cdef class PyPythonKernel:
    cpdef void forward(self, list inputs, cnp.ndarray output)
    cpdef void backward(self, list inputs, list requiresGrad,
                      cnp.ndarray outerGradient, list outputGradients)

