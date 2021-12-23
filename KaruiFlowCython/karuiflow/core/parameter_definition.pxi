cdef class Parameter(Tensor):
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
