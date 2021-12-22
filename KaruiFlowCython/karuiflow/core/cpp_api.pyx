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


cdef class Tensor:

    @staticmethod
    cdef Tensor from_pointer(CppTensor* tensor):
        cdef Tensor obj = Tensor.__new__(Tensor)
        obj.tensor = tensor
        return obj

    def __init__(self, data: np.ndarray, requires_grad=False):
        cdef:
            vector[int] shape = data.shape
            float[:] float_data
            int[:] int_data
            CppTensor* tensor

        if data.dtype == np.float32:
            float_data = data.reshape(-1)
            self.tensor = toTensor(<float *> &float_data[0], shape, <bool> requires_grad)
        elif data.dtype == np.int32:
            int_data = data.reshape(-1)
            self.tensor = toTensor(<int *> &int_data[0], shape, <bool> requires_grad)
        else:
            raise RuntimeError('Unknown data type.')

    def backward(self):
        cdef:
            Storage* storage = self.tensor.getDataStorage()
            Storage* outerGradient = storage.createSimilar()
            float[:] float_ones = np.ones(outerGradient.getSize(), dtype='float32')
        outerGradient.copyFrom(<void*>&float_ones[0])
        self.tensor.backward(outerGradient)
        print('Finished calling backward.')

    def numpy(self):
        cdef:
            cnp.ndarray arr = np.empty(dtype=self.dtype, shape=self.shape)
            void* arr_p = get_pointer(arr, self.dtype.decode("utf-8"))
        self.tensor.copyTo(arr_p)
        return arr

    @property
    def grad(self):
        cdef:
            cnp.ndarray arr = np.empty(dtype=self.dtype, shape=self.shape).reshape(-1)
            void* arr_p = get_pointer(arr, self.dtype.decode("utf-8"))
        self.tensor.copyGradientTo(arr_p)
        return arr.reshape(self.shape)

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

    def __repr__(self):
        return f'' \
               f'Tensor(dtype={self.dtype}, shape={self.shape}, ' \
               f'data={str(self.numpy())})'


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


include "python_kernel_definition.pxi"
include "kernels_definition.pxi"
include "operations_definition.pxi"
include "logging_definition.pxi"
