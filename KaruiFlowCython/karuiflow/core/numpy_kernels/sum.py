import numpy as np

from ..cpp_api import PythonKernel, register_numpy_kernel


@register_numpy_kernel('sum')
class SumKernel(PythonKernel):
    def __init__(self, dim):
        self.dim = dim

    def forward(self, inputs: list, output: np.ndarray):
        assert len(inputs) == 1, f'SumKernel.forward / len(inputs) must be 1, but received {len(inputs)}.'
        np.sum(inputs[0], axis=tuple(self.dim), out=output)

    def backward(self, inputs: list, requiresGrad: list,
                      outerGradient: np.ndarray, outputGradients: list):
        assert len(inputs) == 1, f'SumKernel.backward / len(inputs) must be 1, but received {len(inputs)}.'
        assert len(requiresGrad) == 1, f'SumKernel.backward / len(requiresGrad) must be 1, ' \
                                       f'but received {len(requiresGrad)}.'
        assert len(outputGradients) == 1, f'SumKernel.backward / len(outputGradients) must be 1, ' \
                                          f'but received {len(outputGradients)}.'
        if requiresGrad[0]:
            grad0 = np.ones_like(inputs[0])
            np.multiply(grad0, np.expand_dims(outerGradient, axis=self.dim), out=outputGradients[0])

