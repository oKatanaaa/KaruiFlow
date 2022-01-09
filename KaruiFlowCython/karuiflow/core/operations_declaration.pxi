cdef extern from "KaruiFlowCore.h" namespace "karuiflow":
    cdef cppclass Op:
        CppTensor* call "operator()"(vector[CppTensor*] inputs) except +


cdef extern from "KaruiFlowOperations.h" namespace "karuiflow":
    cdef cppclass CppMatMul "karuiflow::MatMul":
        CppMatMul()

    cdef cppclass CppRelu "karuiflow::Relu":
        CppRelu()

    cdef cppclass CppSum "karuiflow::Sum":
        CppSum(vector[int] dim)

    cdef cppclass CppLog "karuiflow::Log":
        CppLog()

    cdef cppclass CppSigmoid "karuiflow::Sigmoid":
        CppSigmoid()

    cdef cppclass CppSoftmax "karuiflow::Softmax":
        CppSoftmax()

    cdef cppclass CppAdd "karuiflow::Add":
        CppAdd()

    cdef cppclass CppMul "karuiflow::Mul":
        CppMul()

    cdef cppclass CppTo "karuiflow::To":
        CppTo(Device* device)

    cdef cppclass CppReshape "karuiflow::Reshape":
        CppReshape(vector[int] newShape)


cdef class PyOp:
    cdef Op* cpp_op
    cdef void set_op(self, Op* op)


cdef class MatMul(PyOp):
    pass


cdef class Relu(PyOp):
    pass


cdef class Sum(PyOp):
    pass


cdef class Log(PyOp):
    pass


cdef class Sigmoid(PyOp):
    pass


cdef class Softmax(PyOp):
    pass


cdef class Add(PyOp):
    pass


cdef class Mul(PyOp):
    pass


cdef class To(PyOp):
    pass


cdef class Reshape(PyOp):
    pass
