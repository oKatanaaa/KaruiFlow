import numpy as np
import logging

from ..cpp_api import PythonKernel, register_numpy_kernel


@register_numpy_kernel('sum')
class SumKernel(PythonKernel):
    def __init__(self, dim):
        self.dim = tuple(dim)
        if len(self.dim) == 0:
            self.dim = None

    def forward(self, inputs: list, output: np.ndarray):
        assert len(inputs) == 1, f'SumKernel.forward / len(inputs) must be 1, but received {len(inputs)}.'
        logging.debug(f'{self.__class__.__name__}.forward / output.shape = {output.shape}')
        np.sum(inputs[0], axis=self.dim, out=output)
        logging.debug(f'{self.__class__.__name__}.forward successful.')

    def backward(self, inputs: list, requiresGrad: list,
                      outerGradient: np.ndarray, outputGradients: list):
        assert len(inputs) == 1, f'SumKernel.backward / len(inputs) must be 1, but received {len(inputs)}.'
        assert len(requiresGrad) == 1, f'SumKernel.backward / len(requiresGrad) must be 1, ' \
                                       f'but received {len(requiresGrad)}.'
        assert len(outputGradients) == 1, f'SumKernel.backward / len(outputGradients) must be 1, ' \
                                          f'but received {len(outputGradients)}.'
        if requiresGrad[0]:
            grad0 = np.ones_like(inputs[0])
            axis = self.dim
            if axis is None:
                axis = list(range(len(grad0.shape)))
            logging.debug(f'{self.__class__.__name__}.backward / outerGradient.shape = {outerGradient.shape}')
            logging.debug(f'{self.__class__.__name__}.backward / broadcast axes = {axis}')
            expanded_outerGrad = np.expand_dims(outerGradient, axis=axis)
            np.multiply(grad0, expanded_outerGrad, out=outputGradients[0])
            logging.debug(f'{self.__class__.__name__}.backward successful.')


