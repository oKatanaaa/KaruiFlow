# distutils: language = c++
# Cython: language_level=3
# Cython: profile=True
# Cython: cpp_locals=True

DEF CIMPORTS = 1

IF CIMPORTS == 1:
    from libcpp.vector cimport vector
    from libcpp.string cimport string
    from libcpp cimport bool

DEF CIMPORTS = 2

cdef class TestClass:
    pass

cdef extern from "KaruiFlowCore.h" namespace "karuiflow":
    cdef cppclass Device:
        string getDeviceName()

    cdef cppclass DeviceCPU:
        DeviceCPU()

    cdef cppclass DeviceCUDA:
        DeviceCUDA()

    cdef cppclass DType:
        string getName()

    cdef cppclass Float32:
        Float32()

    cdef cppclass Int32:
        Int32()

    cdef struct TensorSpecs:
        DType * dtype;
        vector[int] shape;
        Device * device;

    cdef cppclass Storage:
        void setZero();
        Storage* createSimilar()
        vector[int] getShape()
        int getSize()
        int getSizeBytes()
        void* getData()
        DType* getDtype()
        void copyTo(void * data)
        void copyFrom(void * data)


include "tensor_declaration.pxi"
include "parameter_declaration.pxi"
include "python_kernel_declaration.pxi"
include "operations_declaration.pxi"
include "logging_declaration.pxi"
