import numpy as np

from ..cpp_api import PythonKernel, register_numpy_kernel


@register_numpy_kernel('sigmoid')
class SigmoidKernel(PythonKernel):
    def forward(self, inputs: list, output: np.ndarray):
        assert len(inputs) == 1, f'SigmoidKernel.forward / len(inputs) must be 1, but received {len(inputs)}.'
        exp = np.exp(-inputs[0])
        np.divide(1.0, (1.0 + exp), out=output, dtype='float32')

    def backward(self, inputs: list, requiresGrad: list,
                      outerGradient: np.ndarray, outputGradients: list):
        assert len(inputs) == 1, f'SigmoidKernel.backward / len(inputs) must be 1, but received {len(inputs)}.'
        assert len(requiresGrad) == 1, f'SigmoidKernel.backward / len(requiresGrad) must be 1, ' \
                                       f'but received {len(requiresGrad)}.'
        assert len(outputGradients) == 1, f'SigmoidKernel.backward / len(outputGradients) must be 1, ' \
                                          f'but received {len(outputGradients)}.'
        if requiresGrad[0]:
            exp = np.exp(-inputs[0])
            sig = np.divide(1.0, (1.0 + exp), dtype='float32')
            grad0 = sig * (1 - sig)
            np.multiply(grad0, outerGradient, out=outputGradients[0])

