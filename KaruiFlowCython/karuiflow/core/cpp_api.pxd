from libcpp.vector cimport vector


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
