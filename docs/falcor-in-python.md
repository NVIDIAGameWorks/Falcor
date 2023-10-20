### [Index](./index.md) | Falcor In Python

--------

# Falcor In Python

## Introduction

Falcor has supported Python scripting for a long time. However, scripts have always
been executed in the embedded interpreter in form of either _render scripts_
(creating render graphs, high-level scripting of Mogwai) or through `.pyscene`
files for scene construction.

Recent versions of Falcor can also be used from Python directly by using the
_Falcor Python extension_. This enables Falcor to leverage the power of the
entire Python ecosystem, combining it with popular frameworks such as
[NumPy](https://numpy.org/) or [PyTorch](https://pytorch.org/).
The current implementation is still fairly limited and cannot access all of
Falcor's functionality, but enough API surface is exposed to enable first
machine learning applications.

## Python Environment

To use the _Falcor Python extension_, we need to setup a Python environment.
We suggest to use [Conda](https://docs.conda.io/) for that, because it is
supported on both Windows and Linux and is a popular choice for managing
Python virtual environments (especially on Window).

To setup the environment, download the latest Python 3.10 version of Miniconda
from https://docs.conda.io/projects/miniconda/en/latest/miniconda-other-installer-links.html

Note: The Python version is important here. Falcor ships with a prebuilt binary
version of Python that is used during the build. As long as the same version is
used in the Python environment, there should be binary compatibility. If you
want to build Falcor for a different Python version, set the
`FALCOR_USE_SYSTEM_PYTHON` CMake variable to `ON` and make sure CMake can find
the Python binaries to be used (i.e. run CMake configure within a shell that has
the Python environment activated).

After installation, start the newly installed Miniconda terminal.

To install a basic Falcor/NumPy/PyTorch environment, you can setup the
`falcor-pytorch` environment from the [environment.yml](/environment.yml) file
in the root directory of this repository:

```
conda env create -f environment.yml
```

After installation you can _activate_ the `falcor-pytorch` environment using:

```
conda activate falcor-pytorch
```

To make the _Falcor Python extension_ available in the environment, switch to
the binary output directory of Falcor (i.e. `build/windows-ninja-msvc/bin/Release`)
and run `setpath.bat` (or `setpath.ps1` for PowerShell, `setpath.sh` on Linux).
This will setup the Python and binary paths to allow loading the extension.

With all that you should be able to successfully load the `falcor` module in
a the Python interpreter:

```
>>> import falcor
(Info) Loaded 49 plugin(s) in 0.19s
>>>
```

You can also run the [examples](#examples) listed at the end of this document.

## IDE (Integrated Development Environment) Support

The Falcor build system generates _Python interface stub files_ for the Python
extension, which contains typing information for all the exported Python bindings.
This typing information enables code completion among other things for much improved
developer experience.

We suggest to use VS Code for writing Python code, as it is very capable IDE
for both Python and C++ development. In order to have VS Code operate in the
Python environment we set up in the previous section, it is best to start it
from within the Miniconda terminal by running `code`. This will open VS Code
with all the Python/system paths already set up.

## Examples

There are a few examples available to illustrate the basics of how to work with
Falcor from Python:

- [scripts/python/balls/balls.py](/scripts/python/balls/balls.py):
This illustrates how to use compute shaders from Python and render some moving
2D circles to the screen.
- [scripts/python/gaussian2d/gaussian2d.py](/scripts/python/gaussian2d/gaussian2d.py):
This illustrates how to use differentiable Slang for fitting 2d gaussians to
represent a 2d image. Slang is used to implement a differentiable 2d gaussian
renderer, implementing the backwards pass using Slang's auto differentiation.
The example also illustrates PyTorch/CUDA interop by running the optimization
loop from PyTorch.

## Agility SDK Limitations

Falcor uses Microsoft Agility SDK to get access to the latest features in D3D12.
To make Agility SDK work within the Falcor Python extension make sure to:
- Enable "Developer Mode" in the "For developers" configuration panel in Windows.
- Make sure the Python interpreter executable is installed on the same physical
  drive as the Falcor Python extension.
