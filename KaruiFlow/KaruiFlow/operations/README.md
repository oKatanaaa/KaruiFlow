## CPU Kernels

Every CPU kernel uses Numpy to perform computation. When a CPUKernel class is instantiated, it
calls an extern function defined in Cython to receive a pointer to a Python object that implements
the required functionality.

Though CPU kernel of a single class can be instantiated multiple times, every instance of it 
will receive a pointer to the same Python object. That is because the corresponding Python objects are
created once and are stored in a special container to prevent them from being deleted by 
the garbage collector.
