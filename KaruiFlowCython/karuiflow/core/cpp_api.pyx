# distutils: language = c++
# cython: profile=True

DEF CIMPORTS = 1

IF CIMPORTS == 1:
    from libcpp.vector cimport vector
    from libcpp cimport bool
    from libcpp.string cimport string
    cimport numpy as cnp
    import numpy as np

DEF CIMPORTS = 2


NUMPY_KERNEL_CLASSES = {}
NUMPY_KERNEL_INSTANCES = []


class register_numpy_kernel:
    def __init__(self, kernel_name):
        self.kernel_name = kernel_name

    def __call__(self, cls):
        cached_cls = NUMPY_KERNEL_CLASSES.get(self.kernel_name)
        assert cached_cls is None, f'Kernel with this name has already been registered. ' \
                                   f'Name={self.kernel_name}, cls={cached_cls}'
        NUMPY_KERNEL_CLASSES[self.kernel_name] = cls


def add_numpy_kernel(kernel):
    NUMPY_KERNEL_INSTANCES.append(kernel)


def get_numpy_kernels():
    return NUMPY_KERNEL_INSTANCES

ADD_OP_CLASS = None

def register_add_op(cls):
    global ADD_OP_CLASS
    ADD_OP_CLASS = cls

def get_add_op():
    assert ADD_OP_CLASS is not None, 'No add op has been registered.'
    return ADD_OP_CLASS()

# Transfers tensor from one device to another
TO_OP_CLASS = None

def register_to_op(cls):
    global TO_OP_CLASS
    TO_OP_CLASS = cls

def get_to_op(device):
    assert TO_OP_CLASS is not None, 'No to op has been registered.'
    assert isinstance(device, str), f'Device name must be string, but received {type(device)}'
    return TO_OP_CLASS(device=device)

MUL_OP_CLASS = None

def register_mul_op(cls):
    global MUL_OP_CLASS
    MUL_OP_CLASS = cls

def get_mul_op():
    assert MUL_OP_CLASS is not None, 'No mul op has been registered.'
    return MUL_OP_CLASS()


include "tensor_definition.pxi"
include "parameter_definition.pxi"
include "python_kernel_definition.pxi"
include "kernels_definition.pxi"
include "operations_definition.pxi"
include "logging_definition.pxi"
