cdef extern from "KaruiFlowCore.h" namespace "karuiflow":
    cdef cppclass CppParameter "karuiflow::Parameter":
        CppParameter(CppTensor * tensor)
        void assign(CppTensor * tensor)
        void assignAdd(CppTensor * tensor)


cdef class Parameter(Tensor):
    cdef CppParameter* parameter

    @staticmethod
    cdef Parameter from_pointer(CppParameter * parameter)