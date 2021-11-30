import numpy as np

cimport numpy as cnp

from ..python_kernel cimport _PythonKernel


cdef class MatmulKernel(_PythonKernel):
    cdef void forward(list inputs, cnp.ndarray output):
        np.dot(inputs[0], inputs[1], out=output)
