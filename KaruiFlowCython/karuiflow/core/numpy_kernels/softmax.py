import numpy as np

from ..cpp_api import PythonKernel, register_numpy_kernel


@register_numpy_kernel('softmax')
class SoftmaxKernel(PythonKernel):
    def forward(self, inputs: list, output: np.ndarray):
        assert len(inputs) == 1, f'SoftmaxKernel.forward / len(inputs) must be 1, but received {len(inputs)}.'
        exp = np.exp(inputs[0])
        return np.divide(exp, np.sum(exp, axis=-1, keepdims=True), out=output, dtype='float32')

    def backward(self, inputs: list, requiresGrad: list,
                      outerGradient: np.ndarray, outputGradients: list):
        assert len(inputs) == 1, f'SoftmaxKernel.backward / len(inputs) must be 1, but received {len(inputs)}.'
        assert len(requiresGrad) == 1, f'SoftmaxKernel.backward / len(requiresGrad) must be 1, ' \
                                       f'but received {len(requiresGrad)}.'
        assert len(outputGradients) == 1, f'SoftmaxKernel.backward / len(outputGradients) must be 1, ' \
                                          f'but received {len(outputGradients)}.'
        if requiresGrad[0]:
            exp = np.exp(inputs[0])
            softmax = np.divide(exp, np.sum(exp, axis=-1, keepdims=True), dtype='float32')
            jacobian = softmax[:, None] * (np.identity(softmax.size) - softmax[None, :])
            np.dot(outerGradient, jacobian, out=outputGradients[0])
