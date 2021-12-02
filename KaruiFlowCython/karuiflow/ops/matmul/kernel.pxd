cimport numpy as cnp

from karuiflow.core.python_kernel cimport _PythonKernel


cdef cppclass MatmulKernel(_PythonKernel):
    cdef void pyforward(list inputs, cnp.ndarray output)
