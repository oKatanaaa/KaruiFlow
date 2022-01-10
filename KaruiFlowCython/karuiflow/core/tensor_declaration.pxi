cdef extern from "KaruiFlowCore.h" namespace "karuiflow":
    cdef struct TensorSpecs:
        DType * dtype;
        vector[int] shape;
        Device * device;

    cdef cppclass CppTensor "karuiflow::Tensor":
        CppTensor* clone()
        TensorSpecs getTensorSpecs()
        void setRequiresGrad(bool requiresGrad)
        bool requiresGrad()
        bool isLeaf()

        void backward(Storage* outerGrad) except +
        void zeroGradient() except +

        Storage* getDataStorage()
        Storage* getGradientStorage()
        CppTensor* getGradient() except +

        void copyGradientTo(void* data) except +
        void copyTo(void* data)
        void copyFrom(void* data)
        void to(Device* device) except +

        void decRefCount()


cdef extern from "TensorUtils.h" namespace "karuiflow":
    CppTensor* toTensor(float* data, vector[int] shape, bool requiresGrad)
    CppTensor* toTensor(int* data, vector[int] shape, bool requiresGrad)


cdef class Tensor:
    cdef CppTensor* tensor

    cdef CppTensor* get_cpp_pointer(self)
    @staticmethod
    cdef Tensor from_pointer(CppTensor * tensor)
