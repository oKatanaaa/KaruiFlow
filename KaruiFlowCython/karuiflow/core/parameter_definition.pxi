cdef class Parameter(Tensor):
    @staticmethod
    cdef Parameter from_pointer(CppParameter* parameter):
        cdef Parameter obj = Parameter.__new__(Parameter)
        obj.parameter = parameter
        obj.tensor = <CppTensor*>parameter
        return obj

    @staticmethod
    def from_tensor(Tensor tensor):
        cdef CppTensor* _tensor = tensor.get_cpp_pointer()
        return Parameter.from_pointer(new CppParameter(_tensor))

    def __init__(self, data: np.ndarray, requires_grad=True):
        super().__init__(data=data, requires_grad=requires_grad)
        self.parameter = new CppParameter(self.tensor)
        del self.tensor
        self.tensor = <CppTensor*>self.parameter

    def __iadd__(self, other):
        assert isinstance(other, Parameter) or isinstance(other, Tensor), 'Expected Parameter or Tensor, but ' \
                                                                          f'received {other}'
        cdef Tensor tensor = other
        self.parameter.assignAdd(tensor.get_cpp_pointer())
        return self

    def assign(self, other):
        assert isinstance(other, Parameter) \
               or isinstance(other, Tensor), f'Expected Parameter or Tensor, but received {other}'
        cdef Tensor tensor = other
        self.parameter.assign(tensor.get_cpp_pointer())
        return self
