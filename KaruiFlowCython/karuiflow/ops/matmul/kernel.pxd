cimport numpy as cnp

from ..python_kernel cimport PythonKernel

cdef cppclass MatmulKernel(PythonKernel):
    cdef void pyforward(list inputs, cnp.ndarray output)
