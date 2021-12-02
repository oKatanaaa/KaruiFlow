from .cpp_api cimport PythonOp, PythonKernel


cdef class PyPythonOp:
    cdef PythonOp* python_op

    cdef PythonKernel* instantiate_kernel(self, list dtypes, list shapes, str device)
    cdef tuple infer_output_tensor_specs(self, list dtypes, list shapes, str device)
