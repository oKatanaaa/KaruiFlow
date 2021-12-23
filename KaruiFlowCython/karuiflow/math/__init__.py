from karuiflow.core import MatMul, Softmax, Sigmoid, Sum, Log, Relu, save_op
from karuiflow.core import astensor


def matmul(a, b):
    a = astensor(a)
    b = astensor(b)
    return save_op(MatMul())([a, b])


def softmax(a, dim=None):
    a = astensor(a)
    return save_op(Softmax())([a])


def sigmoid(a):
    a = astensor(a)
    return save_op(Sigmoid())([a])


def sum(a, dim):
    a = astensor(a)
    return save_op(Sum(dim=dim))([a])


def log(a):
    a = astensor(a)
    return save_op(Log())([a])


def relu(a):
    a = astensor(a)
    return save_op(Relu())([a])

