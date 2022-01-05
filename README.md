## Description

This repository contains a university assignment for High Performance Computation course. 

In this assignment we implemented a simple copy of PyTorch deep learning library using Numpy and CUDA.
The following features are supported:
1. Dynamic graphs with backward propagation.
2. ReLU, Log, Addition (limited support), Multiplication (limited support).
3. Batched matrix multiplication (in development).
4. Softmax.
5. Summation of tensors along several axes (only forward pass at the moment).
6. SGD optimizer.
7. Module class for making neural network modules.

Everything is wrapped into a Python API using Cython and corresponds (almost) to PyTorch API, so please use PyTorch documentation as a reference for functionality available in this project.

Folders:
- Architecture - contains several pictures depicting overall architecture of the project.
- KaruiFlow - the core code written in C++ and CUDA.
- KaruiFlowCython - Cython wrapper for the C++ code.

## Installation

The project has several dependencies:
1. CUDA 11.2.
2. CUBLAS.
3. CuTensor.

All of the dependencies are located in `KaruiFlow/dependencies` folder. You have to download and put corresponding `.lib` files into the `lib` folders.

The project is compiled via MVSC 2019. First compile KaruiFlow solution, then run `pip install -e .` in the KaruiFlowCython directory. At this point you're good to go. 

## Current state

The project is still in development and has a very limited set of features. Most of the features are implemented in Numpy only at the moment.
