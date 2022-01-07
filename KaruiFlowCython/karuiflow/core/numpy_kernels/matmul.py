import numpy as np
import logging

from ..cpp_api import PythonKernel, register_numpy_kernel


def transpose_mat(mat):
    # Reverses only the two last dimensions of `mat`
    dims = list(range(len(mat.shape)))
    last, pred_last = dims[-1], dims[-2]
    dims[-1] = pred_last
    dims[-2] = last
    return np.transpose(mat, axes=dims)


@register_numpy_kernel('matmul')
class MatMulKernel(PythonKernel):
    def forward(self, inputs: list, output: np.ndarray):
        assert len(inputs) == 2, f'{self.__class__.__name__}.forward / len(inputs) must be 2, but received {len(inputs)}.'
        A, B = inputs
        logging.debug(f'{self.__class__.__name__}.forward / output.shape = {output.shape}')
        np.matmul(A, B, out=output)
        logging.debug(f'{self.__class__.__name__}.forward successful.')


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
            B_T = transpose_mat(B)
            np.matmul(outerGradient, B_T, out=A_grad)

        if B_requires_grad:
            A_T = transpose_mat(A)
            np.matmul(A_T, outerGradient, out=B_grad)
        logging.debug(f'{self.__class__.__name__}.backward successful.')

