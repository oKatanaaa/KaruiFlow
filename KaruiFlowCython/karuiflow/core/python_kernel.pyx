# distutils: language = c++

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
cimport numpy as cnp
cimport cpython.ref as cpy_ref

from .cpp_api cimport Storage, PythonKernel, DType

import numpy as np


cdef cnp.ndarray float_buff(void* data, vector[int] shape, int n_elems):
    cdef:
        list _shape = <list>shape
        cnp.ndarray buff = np.asarray(<float[:n_elems]>data)
    return np.reshape(buff, shape=_shape)

cdef cnp.ndarray int_buff(void* data, vector[int] shape, int n_elems):
    cdef:
        list _shape = <list>shape
        cnp.ndarray buff = np.asarray(<int[:n_elems]>data)
    return np.reshape(buff, shape=_shape)

cdef cnp.ndarray convert_storage_to_numpy(Storage* storage):
    cdef:
        string dtype = storage.getDtype().getName()
        int n_elems = storage.getSize()
        void* data = storage.getData()
        vector[int] shape = storage.getShape()
        cnp.ndarray buff

    if dtype == b'float32':
        buff = float_buff(data, shape, n_elems)
    elif dtype == b'int32':
        buff = int_buff(data, shape, n_elems)

    return buff

cdef list convert_storages_to_numpy(vector[Storage*] storages):
    cdef:
        list np_storages = []
        Storage* _storage
        cnp.ndarray np_buff

    for _storage in storages:
        np_buff = convert_storage_to_numpy(_storage)
        np_storages.append(np_buff)

    return np_storages


cdef public api:
    void callPyForward(object obj, vector[Storage*] inputs, Storage* output):
        cdef:
            list _inputs = convert_storages_to_numpy(inputs)
            cnp.ndarray np_output = convert_storage_to_numpy(output)

        method = getattr(obj, "forward")
        method(obj, _inputs, np_output)

    void callPyBackward(object obj, vector[Storage *] inputs, vector[bool] requiresGrad,
                      Storage * outerGradient, vector[Storage *] outputGradients):
        cdef:
            list _inputs = convert_storages_to_numpy(inputs)
            list _requiresGrad = <list>requiresGrad
            cnp.ndarray _outerGradient = convert_storage_to_numpy(outerGradient)
            list _outputGradients = convert_storages_to_numpy(outputGradients)
        method = getattr(obj, "backward")
        method(_inputs, _requiresGrad, _outerGradient, _outputGradients)


cdef class PyPythonKernel:
    cpdef void forward(self, list inputs, cnp.ndarray output):
        raise RuntimeError("forward method not implemented.")

    cpdef void backward(self, list inputs, list requiresGrad,
                      cnp.ndarray outerGradient, list outputGradients):
        raise RuntimeError("backward method not implemented.")
