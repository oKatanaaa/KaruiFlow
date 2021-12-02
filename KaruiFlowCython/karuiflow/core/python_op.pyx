from libcpp.vector cimport vector

from .cpp_api cimport TensorSpecs, DType, Float32, Int32



cdef public api:
    PythonKernel* callPyInstantiateKernel(object obj, vector[TensorSpecs] inputs):
        cdef:
            list _inputs = []
            TensorSpecs specs
            str dtype
            list shape
            str device

        for specs in inputs:
            dtype = specs.dtype.getName()
            shape = specs.shape
            device = specs.device.getDeviceName()
            _inputs.append((dtype, shape, device))

        method = getattr(obj, "instantiate_kernel")
        return <PythonKernel*>method(obj, _inputs)

    TensorSpecs callPyInferOutputTensorSpecs(object obj, vector[TensorSpecs] inputs):
        cdef:
            list _inputs = []
            TensorSpecs specs
            str dtype
            list shape
            str device

        for specs in inputs:
            dtype = specs.dtype.getName()
            shape = specs.shape
            device = specs.device.getDeviceName()
            _inputs.append((dtype, shape, device))

        method = getattr(obj, "infer_output_tensor_specs")

        cdef:
            DType* dtype_out
            vector[int] shape_out

        _dtype_out, _shape_out = method(obj, _inputs)
        shape_out = _shape_out

        if _dtype_out == 'float32':
            dtype_out = <DType*>(new Float32())
        elif _dtype_out == 'int32':
            dtype_out = <DType*>(new Int32())
        else:
            raise RuntimeError('Unknown data type:', _dtype_out)

        return TensorSpecs(dtype_out, shape_out, inputs[0].device)
