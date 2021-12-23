import numpy as np

from .cpp_api import Tensor


def astensor(a):
    if isinstance(a, Tensor):
        return a
    elif isinstance(a, np.ndarray):
        return Tensor(a)
    elif isinstance(a, list):
        a = np.asarray(a)
        return Tensor(a)
