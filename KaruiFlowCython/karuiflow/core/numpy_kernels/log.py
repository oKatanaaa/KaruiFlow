import numpy as np
import logging

from ..cpp_api import PythonKernel, register_numpy_kernel


@register_numpy_kernel('log')
class LogKernel(PythonKernel):
    def forward(self, inputs: list, output: np.ndarray):
        assert len(inputs) == 1, f'LogKernel.forward / len(inputs) must be 1, but received {len(inputs)}.'
        logging.debug(f'{self.__class__.__name__}.forward / output.shape = {output.shape}')
        np.log(inputs[0], out=output)

    def backward(self, inputs: list, requiresGrad: list,
                      outerGradient: np.ndarray, outputGradients: list):
        assert len(inputs) == 1, f'LogKernel.backward / len(inputs) must be 1, but received {len(inputs)}.'
        assert len(requiresGrad) == 1, f'LogKernel.backward / len(requiresGrad) must be 1, ' \
                                       f'but received {len(requiresGrad)}.'
        assert len(outputGradients) == 1, f'LogKernel.backward / len(outputGradients) must be 1, ' \
                                          f'but received {len(outputGradients)}.'
        if requiresGrad[0]:
            np.multiply(1. / (inputs[0] + 1e-3), outerGradient, out=outputGradients[0])
        logging.debug(f'{self.__class__.__name__}.backward successful.')
