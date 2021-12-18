from libcpp.vector cimport vector

cdef class PyOp:
    def __cinit__(self, *args, **kwargs):
        pass

    cdef void set_op(self, Op* op):
        self.cpp_op = op

    def __call__(self, list tensors):
        cdef:
            vector[CppTensor *] input_tensors
            Tensor py_tensor
            CppTensor * input_tensor
            CppTensor * output_tensor

        for tensor in tensors:
            py_tensor = tensor
            assert isinstance(tensor, Tensor)
            input_tensor = py_tensor.get_cpp_pointer()
            input_tensors.push_back(input_tensor)
        output_tensor = self.cpp_op.call(input_tensors)
        return Tensor.from_pointer(output_tensor)


cdef class MatMul(PyOp):
    def __cinit__(self, *args, **kwargs):
        self.set_op(<Op*>(new CppMatMul()))


cdef class Relu(PyOp):
    def __cinit__(self, *args, **kwargs):
        self.set_op(<Op*>(new CppRelu()))


cdef class Sum(PyOp):
    def __cinit__(self,  *args, **kwargs):
        cdef list dim = kwargs['dim']
        self.set_op(<Op *> (new CppSum(dim)))


cdef class Log(PyOp):
    def __cinit__(self, *args, **kwargs):
        self.set_op(<Op *> (new CppLog()))


cdef class Sigmoid(PyOp):
    def __cinit__(self, *args, **kwargs):
        self.set_op(<Op *> (new CppSigmoid()))


cdef class Softmax(PyOp):
    def __cinit__(self, *args, **kwargs):
        self.set_op(<Op *> (new CppSoftmax()))
