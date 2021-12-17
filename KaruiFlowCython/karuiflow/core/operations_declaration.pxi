cdef extern from "KaruiFlowOperations.h" namespace "karuiflow":
    cdef cppclass CppMatMul "karuiflow::MatMul":
        CppMatMul()

cdef extern from "KaruiFlowCore.h" namespace "karuiflow":
    cdef cppclass Op:
        CppTensor* call "operator()"(vector[CppTensor*] inputs) except +


cdef class PyOp:
    cdef Op* cpp_op
    cdef void set_op(self, Op* op)


cdef class MatMul(PyOp):
    pass

