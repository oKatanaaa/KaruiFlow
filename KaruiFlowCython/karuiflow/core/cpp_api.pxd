# distutils: language = c++
# Cython: language_level=3
# cython: profile=True

DEF CIMPORTS = 1

IF CIMPORTS == 1:
    from libcpp.vector cimport vector
    from libcpp.string cimport string
    from libcpp cimport bool

DEF CIMPORTS = 2


cdef extern from "KaruiFlowCore.h" namespace "karuiflow":

    cdef cppclass Device:
        string getDeviceName()

    cdef cppclass DeviceCPU:
        void allocateMemory(void **ptr, size_t bytes)
        void deallocateMemory(void* ptr)
        void copyDeviceToCpu(void* src, void* dst, size_t bytes)
        void copyCpuToDevice(void* src, void* dst, size_t bytes)
        void copyDeviceToDevice(void* src, void* dst, size_t bytes)

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

    cdef cppclass CppTensor "karuiflow::Tensor":
        TensorSpecs getTensorSpecs()
        void setRequiresGrad(bool requiresGrad)
        bool requiresGrad()

        void backward(Storage* outerGrad) except +

        Storage* getDataStorage()
        Storage* getGradientStorage()

        void copyGradientTo(void* data) except +
        void copyTo(void* data)
        void copyFrom(void* data)


cdef extern from "TensorUtils.h" namespace "karuiflow":
    CppTensor* toTensor(float* data, vector[int] shape, bool requiresGrad)
    CppTensor* toTensor(int* data, vector[int] shape, bool requiresGrad)


cdef class Tensor:
    cdef CppTensor* tensor

    cdef CppTensor* get_cpp_pointer(self)
    @staticmethod
    cdef Tensor from_pointer(CppTensor * tensor)

include "python_kernel_declaration.pxi"
include "operations_declaration.pxi"
include "logging_declaration.pxi"
