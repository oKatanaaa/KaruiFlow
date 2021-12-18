import numpy as np

from ..cpp_api import PythonKernel, register_numpy_kernel


@register_numpy_kernel('relu')
class ReluKernel(PythonKernel):
    def forward(self, inputs: list, output: np.ndarray):
        assert len(inputs) == 1, f'ReluKernel.forward / len(inputs) must be 1, but received {len(inputs)}.'
        out = np.where(inputs[0] > 0.0, inputs[0], 0.0)
        np.copyto(output, out)

    def backward(self, inputs: list, requiresGrad: list,
                      outerGradient: np.ndarray, outputGradients: list):
        assert len(inputs) == 1, f'ReluKernel.backward / len(inputs) must be 1, but received {len(inputs)}.'
        assert len(requiresGrad) == 1, f'ReluKernel.backward / len(requiresGrad) must be 1, ' \
                                       f'but received {len(requiresGrad)}.'
        assert len(outputGradients) == 1, f'ReluKernel.backward / len(outputGradients) must be 1, ' \
                                          f'but received {len(outputGradients)}.'
        if requiresGrad[0]:
            grad_mask = np.where(inputs[0] > 0.0, 1.0, 0.0)
            np.multiply(outerGradient, grad_mask, out=outputGradients[0])

