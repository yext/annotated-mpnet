#!/usr/bin/env python3

"""
Setup script for the annotated_mpnet library
"""

from setuptools import setup, find_packages, Extension
import sys

# Exit if running Python2
if sys.version_info < (3,):
    sys.exit("Python3 required to install and run annotated_mpnet")

# Open up the readme to be loaded into the setup function below
with open("README.md") as f:
    readme = f.read()

# Create the necessary Cython extension for our fast perm utils

# Have to include some additional compilation args depending on platfom
if sys.platform == "darwin":
    extra_compile_args = ["-stdlib=libc++", "-O3"]
else:
    extra_compile_args = ["-std=c++11", "-O3"]

# We need to build a subclass of Extension to get the numpy extensions to install properly.
# Otherwise, install will fail in envs that don't have numpy previously installed
class NumpyExtension(Extension):
    """Source: https://stackoverflow.com/a/54128391"""

    def __init__(self, *args, **kwargs):
        self.__include_dirs = []
        super().__init__(*args, **kwargs)

    @property
    def include_dirs(self):
        import numpy

        return self.__include_dirs + [numpy.get_include()]

    @include_dirs.setter
    def include_dirs(self, dirs):
        self.__include_dirs = dirs


# Now make reference to the specific extension
extensions = [
    NumpyExtension(
        "annotated_mpnet.utils.perm_utils_fast",
        sources=["annotated_mpnet/utils/perm_utils_fast.pyx"],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    name="annotated_mpnet",
    version="0.1.0",
    description="Raw Torch, heavily annotated, pretrainable MPNet",
    url="https://github.com/yext/annotated-mpnet",
    long_description=readme,
    long_description_content_type="text/markdown",
    setup_requires=[
        "cython",
        "numpy",
        "setuptools>=18.0",
    ],
    install_requires=["cython", "numpy", "rich", "torch", "transformers"],
    packages=find_packages(exclude=["cli_tools", "tests"]),
    ext_modules=extensions,
    test_suite="tests",
    entry_points={
        "console_scripts": [
            "pretrain-mpnet = cli_tools.pretrain_mpnet:cli_main",
            "convert-to-hf = cli_tools.convert_pretrained_mpnet_to_hf_model:cli_main",
        ]
    },
)
