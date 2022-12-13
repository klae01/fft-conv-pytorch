#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pipenv install twine --dev

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command, Extension
from torch.utils import cpp_extension


# Package meta-data.
NAME = "fft-conv-pytorch"
DESCRIPTION = "Implementation of 1D, 2D, and 3D FFT convolutions in PyTorch."
URL = "https://github.com/klae01/fft-conv-pytorch"
EMAIL = "tspt2479@gmail.com"
AUTHOR = "klae01"
REQUIRES_PYTHON = ">=3.7.0"
VERSION = None  #'0.0.0'
project_slug = None

if project_slug is None:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")

# What packages are required for this module to be executed?
REQUIRED = [
    "numba",
    "numpy",
    "torch>=1.8",
]

# What packages are optional?
EXTRAS = {"test": ["black", "flake8", "isort", "pytest", "pytest-cov"]}

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    with open(os.path.join(here, project_slug, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION


class UploadCommand(Command):
    """Support setup.py upload."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(here, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPI via Twine…")
        os.system("twine upload dist/*")

        self.status("Pushing git tags…")
        os.system("git tag v{0}".format(about["__version__"]))
        os.system("git push --tags")

        sys.exit()


ext_modules = [
    cpp_extension.CppExtension(
        "fft_conv_pytorch.optimized.lib.cpu",
        ["fft_conv_pytorch/optimized/lib/MemoryAccessCost.cpp"],
        extra_compile_args={"cxx": ["-O3"]},
    ),
    cpp_extension.CppExtension(
        "fft_conv_pytorch.optimized.lib.cpu",
        ["fft_conv_pytorch/optimized/lib/PlaneDot.cpp"],
        extra_compile_args={"cxx": ["-O3"]},
    ),
    cpp_extension.CppExtension(
        "fft_conv_pytorch.optimized.lib.cuda",
        ["fft_conv_pytorch/optimized/lib/PlaneDot.cu"],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": ["-O3"],
        },
    ),
]

# Where the magic happens:
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    ext_modules=ext_modules,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],
    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="MIT",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: C++",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
    ],
    # $ setup.py publish support.
    cmdclass={
        "build_ext": cpp_extension.BuildExtension,
        "upload": UploadCommand,
    },
)
