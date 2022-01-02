import numpy as np
import logging

from ..cpp_api import PythonKernel, register_numpy_kernel


def flatten(x):
    dim = x.shape[-1]
    return x.reshape(-1, dim)


@register_numpy_kernel('softmax')
class SoftmaxKernel(PythonKernel):
    def forward(self, inputs: list, output: np.ndarray):
        assert len(inputs) == 1, f'SoftmaxKernel.forward / len(inputs) must be 1, but received {len(inputs)}.'
        logging.debug(f'{self.__class__.__name__}.forward / output.shape = {output.shape}')
        logits = inputs[0] - np.max(inputs[0], axis=-1, keepdims=True)
        exp = np.exp(logits)
        np.divide(exp, np.sum(exp, axis=-1, keepdims=True), out=output, dtype='float32')
        logging.debug(f'{self.__class__.__name__}.forward successful.')

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
            softmax = flatten(softmax)
            outer_grad = flatten(outerGradient)
            outer_grad = np.expand_dims(outer_grad, axis=1)

            jacobian = np.expand_dims(softmax, axis=1) * (np.identity(softmax.shape[-1])[None, ...] - softmax[..., None])
            grad = np.matmul(outer_grad, jacobian)
            np.copyto(outputGradients[0].reshape(-1), grad.reshape(-1))
            logging.debug(f'{self.__class__.__name__}.backward successful.')