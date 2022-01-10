## Description

This repository contains a university assignment for High Performance Computation course. 

In this assignment we implemented a simple copy of PyTorch deep learning library using Numpy and CUDA.
The following features are supported:
- General:
1. Dynamic graphs with backward propagation.
2. SGD optimizer (with momentum).
3. `Module` class for making neural network modules.
- CPU operation:
1. ReLU.
2. Log.
3. Sigmoid.
4. Element-wise addition.
5. Element-wise multiplication.
6. Matrix multiplication.
7. Softmax.
8. Summation of tensors along several axes.
- CUDA operation:
1. ReLU.
2. Log - only float datatype.
3. Sigmoid - only float datatype.
4. Element-wise addition - only float datatype.
5. Element-wise multiplication - only float datatype.
6. Matrix multiplication (only pure matrice, no batch support) - only float datatype.
7. Softmax - only float datatype.
8. Summation of tensors along several axes - only float datatype.

The above lists will be updated over time.


Since it is a "PyTorch"-ish Python library, you can implement arbitrary neural networks using the provided components and train them using gradient descent. You can also use it for arbitrary computational purposes (on GPU). 

Everything is wrapped into a Python API using Cython and corresponds (almost) to PyTorch API, so please use PyTorch documentation as a reference for functionality available in this project.

Folders:
- Architecture - contains several pictures depicting the overall architecture of the project.
- KaruiFlow - the core code written in C++ and CUDA.
- KaruiFlowCython - Cython wrapper for the C++ code.

## Installation

The project has several dependencies:
1. CUDA 11.2.
2. CUBLAS.
3. CuTensor.
4. cuDNN.

All of the dependencies are located in `KaruiFlow/dependencies` folder. You have to download and put corresponding `.lib` files into the `lib` folders.

The project is compiled via MVSC 2019. First compile KaruiFlow solution, then run `pip install -e .` in the KaruiFlowCython directory. At this point you're good to go. 

## Current state

The project is still in development and has a very limited set of features (it is also quite buggy).

## Authors

Igor Kilbas - https://github.com/oKatanaaa (KaruiFlow design)
Yulia Bogdanova - https://github.com/AlexRikka (CUDA kernels)
