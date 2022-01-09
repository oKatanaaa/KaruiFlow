import numpy as np
import logging

from ..cpp_api import PythonKernel, register_numpy_kernel


@register_numpy_kernel('add')
class AddKernel(PythonKernel):
    def forward(self, inputs: list, output: np.ndarray):
        assert len(inputs) == 2, f'{self.__class__.__name__}.forward / len(inputs) must be 2, but received {len(inputs)}.'
        A, B = inputs
        np.add(A, B, out=output)
        logging.debug(f'{self.__class__.__name__}.forward successful.')

    def backward(self, inputs: list, requiresGrad: list,
                      outerGradient: np.ndarray, outputGradients: list):
        assert len(inputs) == 2, f'{self.__class__.__name__}.backward / len(inputs) must be 2, but received {len(inputs)}.'
        assert len(requiresGrad) == 2, f'{self.__class__.__name__}.backward / len(requiresGrad) must be 2, ' \
                                       f'but received {len(requiresGrad)}.'
        assert len(outputGradients) == 2, f'{self.__class__.__name__}.backward / len(outputGradients) must be 2, ' \
                                          f'but received {len(outputGradients)}.'
        A, B = inputs
        A_requires_grad, B_requires_grad = requiresGrad
        # Output buffers
        A_grad, B_grad = outputGradients

        if A_requires_grad:
            dims_to_reduce = []
            for i, dim in enumerate(A.shape):
                if dim == 1:
                    dims_to_reduce.append(i)
            outerGradient_A = np.sum(outerGradient, axis=tuple(dims_to_reduce), keepdims=True)
            np.copyto(A_grad, np.ones_like(A) * outerGradient_A)

        if B_requires_grad:
            dims_to_reduce = []
            for i, dim in enumerate(B.shape):
                if dim == 1:
                    dims_to_reduce.append(i)
            outerGradient_B = np.sum(outerGradient, axis=tuple(dims_to_reduce), keepdims=True)
            np.copyto(B_grad, np.ones_like(B) * outerGradient_B)
        logging.debug(f'{self.__class__.__name__}.backward successful.')
