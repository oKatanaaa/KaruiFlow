# distutils: language = c++

from libcpp.vector cimport vector
from libcpp.string cimport string
cimport cpython.ref as cpy_ref


cdef extern from "KaruiFlow.h" namespace "karuiflow":
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

    cdef cppclass Storage:
        Storage()
        Storage(DType* dtype, vector[int] shape, Device* device)
        vector[int] getShape()
        size_t getSize()
        size_t getSizeBytes()
        DType* getDtype()
        void* getData()

    cdef struct TensorSpecs:
        DType* dtype;
        vector[int] shape;
        Device* device;


cdef extern from "KaruiFlow.h" namespace "karuiflow":
    cdef cppclass PythonKernel:
        PythonKernel()
        PythonKernel(cpy_ref.PyObject* obj)

    cdef cppclass PythonOp:
        PythonKernel * instantiateKernel(vector[TensorSpecs] inputs)
        TensorSpecs inferOutputTensorSpecs(vector[TensorSpecs] inputs)
