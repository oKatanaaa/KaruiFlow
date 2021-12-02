# distutils: language = c++

from libcpp.vector cimport vector
cimport cpython.ref as cpy_ref

# Memory API



cdef class PyDeviceCPU:
    cdef DeviceCPU* _this

    def __cinit__(self):
        self._this = new DeviceCPU()

    def allocateMemory(self, int bytes):
        cdef void* address = NULL
        self._this.allocateMemory(&address, bytes)
        return <int>address

    def deallocateMemory(self, int address):
        self._this.deallocateMemory(<void*>address)

    def __dealloc__(self):
        if self._this != NULL:
            del self._this


cdef class NoneStorage:
    cdef Storage* _this

    def __cinit__(self, device, dtype, list shape):
        cdef:
            DType* _dtype
            vector[int] _shape = <vector[int]>shape
            Device* _device
        if dtype == 'float32':
            _dtype = <DType*>(new Float32())
        elif dtype == 'int32':
            _dtype = <DType*>(new Int32())

        if device == 'cpu':
            _device = <Device*>(new DeviceCPU())
        self._this = new Storage(_dtype, _shape, _device)

    def get_shape(self):
        return self._this.getShape()

    def get_size_bytes(self):
        return self._this.getSizeBytes()
