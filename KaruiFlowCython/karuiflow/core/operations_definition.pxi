from libcpp.vector cimport vector


OPERATION_WRAPPER_INSTANCES = []


def get_operation_wrapper_instances():
    return OPERATION_WRAPPER_INSTANCES


def save_op(op):
    OPERATION_WRAPPER_INSTANCES.append(op)
    return op


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
            assert isinstance(tensor, Tensor), 'PyOp must receive a list of Tensors,' \
                                               f' but received {tensors}'
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
        cdef list dim = list(kwargs['dim'])
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


cdef class Add(PyOp):
    def __cinit__(self, *args, **kwargs):
        self.set_op(<Op *> (new CppAdd()))


cdef class Mul(PyOp):
    def __cinit__(self, *args, **kwargs):
        self.set_op(<Op *> (new CppMul()))


cdef class To(PyOp):
    def __cinit__(self, *args, **kwargs):
        device_name = kwargs["device"]
        cdef Device* device
        if device_name == "cuda":
            device = <Device*>(new DeviceCUDA())
        elif device_name == "cpu":
            device = <Device *> (new DeviceCPU())
        else:
            raise RuntimeError("Unknown device. Expected cuda or cpu, but received " + device_name)
        self.set_op(<Op*>(new CppTo(device)))


cdef class Reshape(PyOp):
    def __cinit__(self, *args, **kwargs):
        cdef list new_shape = list(kwargs['new_shape'])
        self.set_op(<Op *> (new CppReshape(new_shape)))
