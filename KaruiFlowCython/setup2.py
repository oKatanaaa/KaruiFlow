from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
from Cython.Distutils import build_ext

from numpy import get_include
import os
from glob import glob

extensions = Extension(
    '*',
    sources=glob('karuiflow/**/*.pyx'),
    libraries=['KaruiFlow'],
    language='c++',
    include_dirs=['../KaruiFlow/KaruiFlow/core/headers/', get_include()],
    library_dirs=['../KaruiFlow/x64/Release'],
    extra_compile_args=['/openmp']
)

setup(
    name='karuiflow',
    ext_modules=cythonize(extensions),
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    install_requires=[],
    cmdclass={'build_ext': build_ext},
)
