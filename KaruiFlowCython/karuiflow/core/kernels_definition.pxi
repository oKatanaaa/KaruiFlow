IF CIMPORTS == 1:
    from .python_kernel_declaration cimport PyPythonKernel
    import numpy as np

cimport cpython.ref as cpy_ref

cdef public api cpy_ref.PyObject* callPyGetMatMul():
    object = MatMulKernel()
    return <cpy_ref.PyObject*>object

cdef class MatMulKernel(PyPythonKernel):
    cpdef void forward(self, list inputs, cnp.ndarray output):
        assert len(inputs) == 2, f'MatMulKernel.forward / len(inputs) must be 2, but received {len(inputs)}.'
        A, B = inputs
        np.dot(A, B, out=output)

    cpdef void backward(self, list inputs, list requiresGrad,
                      cnp.ndarray outerGradient, list outputGradients):
        assert len(inputs) == 2, f'MatMulKernel.backward / len(inputs) must be 2, but received {len(inputs)}.'
        assert len(requiresGrad) == 2, f'MatMulKernel.backward / len(requiresGrad) must be 2, ' \
                                       f'but received {len(requiresGrad)}.'
        assert len(outputGradients) == 2, f'MatMulKernel.backward / len(outputGradients) must be 2, ' \
                                          f'but received {len(outputGradients)}.'
        A, B = inputs
        A_requires_grad, B_required_grad = requiresGrad
        # Output buffers
        A_grad, B_grad = outputGradients

        if A_requires_grad:
            np.dot(outerGradient, B.T, out=A_grad)

        if B_required_grad:
            np.dot(A.T, outerGradient, out=B_grad)
