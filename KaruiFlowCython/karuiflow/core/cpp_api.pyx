# distutils: language = c++
DEF CIMPORTS = 1

IF CIMPORTS == 1:
    from libcpp.vector cimport vector
    from libcpp cimport bool
    from libcpp.string cimport string
    cimport numpy as cnp
    import numpy as np

DEF CIMPORTS = 2

cdef void* get_pointer(cnp.ndarray arr, str dtype):
    cdef:
        float[:] float_buff
        int[:] int_buff

    if dtype == 'float32':
        float_buff = arr.reshape(-1)
        return <void*>&float_buff[0]
    elif dtype == 'int32':
        int_buff = arr.reshape(-1)
        return <void*>&float_buff[0]


cdef class PyTensor:

    @staticmethod
    cdef PyTensor from_pointer(CppTensor* tensor):
        cdef PyTensor obj = PyTensor.__new__(PyTensor)
        obj.tensor = tensor
        return obj

    def __cinit__(self, data: np.ndarray, requires_grad=False):
        cdef:
            vector[int] shape = data.shape
            float[:] float_data
            int[:] int_data
            CppTensor* tensor

        if data.dtype == np.float32:
            float_data = data.reshape(-1)
            self.tensor = toTensor(<float *> &float_data[0], shape, <bool> requires_grad)
            print('Created float32 tensor.')
        elif data.dtype == np.int32:
            int_data = data.reshape(-1)
            self.tensor = toTensor(<int *> &int_data[0], shape, <bool> requires_grad)
        else:
            raise RuntimeError('Unknown data type.')

    def numpy(self):
        cdef:
            cnp.ndarray arr = np.empty(dtype=self.dtype, shape=self.shape)
            void* arr_p = get_pointer(arr, self.dtype)
        self.tensor.copyTo(arr_p)
        return arr

    @property
    def grad(self):
        cdef:
            cnp.ndarray arr = np.empty(dtype=self.dtype, shape=self.shape).reshape(-1)
            void* arr_p = get_pointer(arr, self.dtype)
        self.tensor.copyGradientTo(arr_p)
        return arr

    @property
    def dtype(self):
        cdef string _dtype = self.tensor.getTensorSpecs().dtype.getName()
        return _dtype

    @property
    def shape(self):
        cdef vector[int] _shape = self.tensor.getTensorSpecs().shape
        return _shape

    @property
    def device(self):
        cdef string device_name = self.tensor.getTensorSpecs().device.getDeviceName()
        return device_name

    cdef CppTensor* get_cpp_pointer(self):
        return self.tensor

NUMPY_KERNELS = []

def add_numpy_kernel(kernel):
    NUMPY_KERNELS.append(kernel)

def get_numpy_kernels():
    return NUMPY_KERNELS

include "python_kernel_definition.pxi"
include "kernels_definition.pxi"
include "operations_definition.pxi"
