import numpy as np

from ..cpp_api import PythonKernel, register_numpy_kernel


@register_numpy_kernel('matmul')
class MatMulKernel(PythonKernel):
    def forward(self, inputs: list, output: np.ndarray):
        assert len(inputs) == 2, f'MatMulKernel.forward / len(inputs) must be 2, but received {len(inputs)}.'
        A, B = inputs
        np.dot(A, B, out=output)

    def backward(self, inputs: list, requiresGrad: list,
                      outerGradient: np.ndarray, outputGradients: list):
        assert len(inputs) == 2, f'MatMulKernel.backward / len(inputs) must be 2, but received {len(inputs)}.'
        assert len(requiresGrad) == 2, f'MatMulKernel.backward / len(requiresGrad) must be 2, ' \
                                       f'but received {len(requiresGrad)}.'
        assert len(outputGradients) == 2, f'MatMulKernel.backward / len(outputGradients) must be 2, ' \
                                          f'but received {len(outputGradients)}.'
        A, B = inputs
        A_requires_grad, B_requires_grad = requiresGrad
        # Output buffers
        A_grad, B_grad = outputGradients

        if A_requires_grad:
            np.dot(outerGradient, B.T, out=A_grad)

        if B_requires_grad:
            np.dot(A.T, outerGradient, out=B_grad)

