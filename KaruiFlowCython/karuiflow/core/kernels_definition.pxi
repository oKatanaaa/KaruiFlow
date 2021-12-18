IF CIMPORTS == 1:
    from .python_kernel_declaration cimport PythonKernel
    from .cpp_api import NUMPY_KERNEL_CLASSES
    import numpy as np

cimport cpython.ref as cpy_ref

cdef cpy_ref.PyObject* instantiate_kernel(str kernel_name):
    kernel = NUMPY_KERNEL_CLASSES[kernel_name]()
    add_numpy_kernel(kernel)
    return <cpy_ref.PyObject *> kernel

cdef public api:
    cpy_ref.PyObject* callPyGetMatMul():
        return instantiate_kernel('matmul')

    cpy_ref.PyObject * callPyGetRelu():
        return instantiate_kernel('relu')

    cpy_ref.PyObject * callPyGetSigmoid():
        return instantiate_kernel('sigmoid')

    cpy_ref.PyObject * callPyGetSoftmax():
        return instantiate_kernel('softmax')

    cpy_ref.PyObject * callPyGetSum(vector[int] dim):
        cdef list _dim = dim
        kernel = NUMPY_KERNEL_CLASSES['sum'](_dim)
        add_numpy_kernel(kernel)
        return <cpy_ref.PyObject *>kernel

    cpy_ref.PyObject * callPyGetLog():
        return instantiate_kernel('log')
