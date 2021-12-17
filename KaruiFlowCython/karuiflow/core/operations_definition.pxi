from libcpp.vector cimport vector

cdef class PyOp:
    def __cinit__(self):
        pass

    cdef void set_op(self, Op* op):
        self.cpp_op = op

    def __call__(self, list tensors):
        cdef:
            vector[CppTensor *] input_tensors
            PyTensor py_tensor
            CppTensor * input_tensor
            CppTensor * output_tensor

        for tensor in tensors:
            py_tensor = tensor
            assert isinstance(tensor, PyTensor)
            input_tensor = py_tensor.get_cpp_pointer()
            input_tensors.push_back(input_tensor)
        print("Calling cpp op")
        output_tensor = self.cpp_op.call(input_tensors)
        print("Constructing new tensor")
        return PyTensor.from_pointer(output_tensor)


cdef class MatMul(PyOp):
    def __cinit__(self):
        self.set_op(<Op*>(new CppMatMul()))

