from setuptools import find_packages
from distutils.extension import Extension
from distutils.core import setup
from Cython.Build import cythonize
from Cython.Distutils import build_ext

from numpy import get_include
import os
from glob import glob

extensions = Extension(
    '*',
    sources=[
        'karuiflow/core/cpp_api.pyx'
    ],
    libraries=['KaruiFlow'],
    language='c++',
    include_dirs=[
        '../KaruiFlow/dependencies/cutensor/11.0/include',
        '../KaruiFlow/dependencies/cuda/11.2/include',
        '../KaruiFlow/KaruiFlow/',
        '../KaruiFlow/KaruiFlow/core/headers/',
        '../KaruiFlow/KaruiFlow/utilities/',
        '../KaruiFlow/KaruiFlow/operations/',
        get_include()],
    library_dirs=['../KaruiFlow/KaruiFlow/x64/Release', '../KaruiFlow/KaruiCUDA/x64/Release'],
    #extra_compile_args=['/openmp']
)

print(find_packages())
setup(
    name='karuiflow',
    ext_modules=cythonize([extensions]),
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    install_requires=[],
    cmdclass={'build_ext': build_ext},
)
