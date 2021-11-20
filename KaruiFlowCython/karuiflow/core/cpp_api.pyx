from libcpp.vector cimport vector

# Memory API
cdef extern from "KaruiFlow.h" namespace "karuiflow":
    cdef cppclass Device:
        pass

    cdef cppclass DeviceCPU:
        void allocateMemory(void **ptr, size_t bytes)
        void deallocateMemory(void* ptr)
        void copyDeviceToCpu(void* src, void* dst, size_t bytes)
        void copyCpuToDevice(void* src, void* dst, size_t bytes)
        void copyDeviceToDevice(void* src, void* dst, size_t bytes)

    cdef cppclass DType:
        pass

    cdef cppclass Float32:
        Float32()

    cdef cppclass Int32:
        Int32()

    cdef cppclass Storage:
        Storage(DType dtype, vector[int] shape, Device* device)

        vector[int] getShape()
        size_t getSize()
        size_t getSizeBytes()
        DType getDtype()
        void* getData()


cdef extern from "KaruiFlow.h" namespace "karuiflow":
    cdef cppclass Kernel:
        void forward(vector[Storage*] inputs, Storage* output)
        vector[Storage] backward(vector[Storage*] inputs, vector[bool] requiresGrad, Storage)


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


cdef class PyStorage:
    cdef Storage* _this

    def __cinit__(self, device, dtype, list shape):
        cdef:
            DType _dtype
            vector[int] _shape = <vector[int]>shape
            Device* _device
        if dtype == 'float32':
            _dtype = <DType>Float32()
        elif dtype == 'int32':
            _dtype = <DType>Int32()

        if device == 'cpu':
            _device = <Device*>(new DeviceCPU())
        self._this = new Storage(_dtype, _shape, _device)

    def get_shape(self):
        return self._this.getShape()

    def get_size_bytes(self):
        return self._this.getSizeBytes()
