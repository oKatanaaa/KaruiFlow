# distutils: language = c++
#

IF CIMPORTS == 1:
    from libcpp.vector cimport vector
    from libcpp.string cimport string
    from libcpp cimport bool
    cimport numpy as cnp
    cimport cpython.ref as cpy_ref

    from .cpp_api cimport Storage

import numpy as np
import logging



cdef cnp.ndarray float_buff(void* data, vector[int] shape, int n_elems):
    cdef:
        list _shape = <list>shape
        cnp.ndarray buff = np.asarray(<float[:n_elems]>data)
    return np.reshape(buff, newshape=_shape)

cdef cnp.ndarray int_buff(void* data, vector[int] shape, int n_elems):
    cdef:
        list _shape = <list>shape
        cnp.ndarray buff = np.asarray(<int[:n_elems]>data)
    return np.reshape(buff, newshape=_shape)

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

cdef list convert_storages_to_numpy(vector[Storage*] storages, list requires_convertation):
    cdef:
        list np_storages = []
        Storage* _storage
        cnp.ndarray np_buff

    for i in range(storages.size()):
        _storage = storages[i]
        _requires_convertation = requires_convertation[i]
        if not _requires_convertation:
            np_storages.append(None)
            continue

        np_buff = convert_storage_to_numpy(_storage)
        np_storages.append(np_buff)
    return np_storages


cdef public:
    void callPyForward(object obj, vector[Storage*] inputs, Storage* output) with gil:
        logging.debug('Received forward call on Python side.')
        cdef:
            list _inputs = convert_storages_to_numpy(inputs, [True] * inputs.size())
            cnp.ndarray np_output = convert_storage_to_numpy(output)
        method = getattr(obj, "forward")
        logging.debug(f'Calling forward in {obj}.')
        method(_inputs, np_output)

    void callPyBackward(object obj, vector[Storage *] inputs, vector[bool] requiresGrad,
                      Storage * outerGradient, vector[Storage *] outputGradients) with gil:
        logging.debug('Received backward call on Python side.')
        cdef:
            list _inputs = convert_storages_to_numpy(inputs, [True] * inputs.size())
            list _requiresGrad = <list>requiresGrad
            cnp.ndarray _outerGradient = convert_storage_to_numpy(outerGradient)
            list _outputGradients = convert_storages_to_numpy(outputGradients, requiresGrad)
        method = getattr(obj, "backward")
        logging.debug(f'Calling backward in {obj}.')
        method(_inputs, _requiresGrad, _outerGradient, _outputGradients)


cdef class PythonKernel:
    def forward(self, list inputs, cnp.ndarray output):
        raise RuntimeError("forward method not implemented.")

    def backward(self, list inputs, list requiresGrad,
                      cnp.ndarray outerGradient, list outputGradients):
        raise RuntimeError("backward method not implemented.")
