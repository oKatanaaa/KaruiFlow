cdef void* get_pointer(cnp.ndarray arr, str dtype):
    cdef:
        float[:] float_buff
        int[:] int_buff

    if dtype == 'float32':
        float_buff = arr.reshape(-1)
        return <void*>&float_buff[0]
    elif dtype == 'int32':
        int_buff = arr.reshape(-1)
        return <void*>&int_buff[0]


cdef class Tensor:
    def __init__(self, data, requires_grad=False):
        assert isinstance(data, np.ndarray), f'Expected np.ndarray, but received {type(data)}'
        cdef:
            vector[int] shape
            float[:] float_data
            int[:] int_data
            CppTensor* tensor

        shape = data.shape
        if data.dtype == np.float32:
            float_data = data.reshape(-1)
            self.tensor = toTensor(<float *> &float_data[0], shape, <bool> requires_grad)
        elif data.dtype == np.int32:
            int_data = data.reshape(-1)
            self.tensor = toTensor(<int *> &int_data[0], shape, <bool> requires_grad)
        else:
            raise RuntimeError(f'Unknown data type. Expected int32 or float32, but received {data}')

    @staticmethod
    cdef Tensor from_pointer(CppTensor * tensor):
        cdef Tensor obj = Tensor.__new__(Tensor)
        obj.tensor = tensor
        return obj


    def backward(self):
        cdef:
            Storage* storage = self.tensor.getDataStorage()
            Storage* outerGradient = storage.createSimilar()
            float[:] float_ones = np.ones(outerGradient.getSize(), dtype='float32')
        outerGradient.copyFrom(<void*>&float_ones[0])
        self.tensor.backward(outerGradient)
        del outerGradient

    def zero_grad(self):
        self.tensor.zeroGradient()

    def numpy(self):
        cdef:
            cnp.ndarray arr = np.empty(dtype=self.dtype, shape=self.shape)
            void* arr_p = get_pointer(arr, self.dtype.decode("utf-8"))
        self.tensor.copyTo(arr_p)
        return arr

    @property
    def grad(self):
        cdef:
            CppTensor* grad
        try:
            grad = self.tensor.getGradient()
        except RuntimeError as r:
            return None
        return Tensor.from_pointer(grad)

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

    @property
    def requires_grad(self):
        return self.tensor.requiresGrad()

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
        assert isinstance(other, (Tensor, float)), f"Can multiply Tensor with Tensor or float, but received {type(other)}"
        if isinstance(other, float):
            other = np.array(other, dtype='float32')
            other = Tensor(other).to(self.device.decode('utf'), inplace=True)
        mul_op = get_mul_op()
        return mul_op([self, other])

    def to(self, str device, bint inplace=False):
        cdef Device* _device
        if inplace:
            if device == 'cuda':
                _device = <Device*>new DeviceCUDA()
            elif device == 'cpu':
                _device = <Device*>new DeviceCPU()
            else:
                raise RuntimeError(f'Expected cuda or cpu, be received {device}')
            self.tensor.to(_device)
            return self

        to_op = get_to_op(device)
        return to_op([self])

    def __dealloc__(self):
        if self.tensor != NULL:
            self.tensor.decRefCount()

