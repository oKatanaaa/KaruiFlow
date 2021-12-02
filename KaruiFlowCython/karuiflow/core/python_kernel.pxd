# distutils: language = c++

cimport numpy as cnp

from .cpp_api cimport PythonKernel


cdef class PyPythonKernel:
    cdef PythonKernel* python_kernel

    cdef void forward(self, list inputs, cnp.ndarray output)
    cdef void backward(self, list inputs, list requiresGrad,
                      cnp.ndarray outerGradient, list outputGradients)
