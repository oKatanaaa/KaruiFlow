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

    @property
    def is_leaf(self):
        return self.tensor.isLeaf()

    def clone(self):
        cdef CppTensor* tensor = self.tensor.clone()
        return Tensor.from_pointer(tensor)

    cdef CppTensor* get_cpp_pointer(self):
        return self.tensor

    def __repr__(self):
        return f'' \
               f'Tensor(dtype={self.dtype}, shape={self.shape}, ' \
               f'data={str(self.numpy())})'

    def __add__(self, other):
        assert isinstance(other, Tensor), f"Can add Tensor only with Tensor, but received {type(other)}"
        add_op = get_add_op()
        return add_op([self, other])

    def __mul__(self, other):
        assert isinstance(other, Tensor), f"Can multiply Tensor only with Tensor, but received {type(other)}"
        mul_op = get_mul_op()
        return mul_op([self, other])
