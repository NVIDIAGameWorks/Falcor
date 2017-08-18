This sample requires Python with TensorFlow installed, as it embeds a Python interpreter (by linking to Python librarires).
Since this is not a regular part of the Falcor distribution, using this sample (and Falcor's built-in PythonEmbedding class) 
requires a bit more work than just building Falcor.

There are 2 parts to this additional setup:
   1) Telling Falcor to compile with Python support 
   2) Installing an appropriate Python tool chain

First, tell Falcor to compile with Python support.
   1) Open FalcorConfig.h (see same directory as Falcor.h), and change the #define FALCOR_USE_PYTHON to be 1
   2) In your OS, define two environment variables FALCOR_PYTHON_PATH and FALCOR_PYBIND11_PATH, depending on install locations.
        * Remember to restart your Visual Studio so it grabs the updated environment variables.
		* Ours paths were set: FALCOR_PYTHON_PATH=C:\Python\Python35 and FALCOR_PYBIND11_PATH=C:\Python\pybind11

Second, install an appropriate Python toolchain.
   1) Install Python.  This needs to be a version compatible with TensorFlow.  
        * We grabbed directly from http://python.org
        * We use Python 3.5.3 (any 3.5.x should work, but 3.6.x is not yet compatible with TensorFlow)
   2) In C++, we interface with Python (mostly) using the pybind11 library.  
        * This is really easy, since it's a header-only library.
        * Grab that from here: https://github.com/pybind/pybind11
   3) If you want GPU-accelerated TensorFlow, install CUDA and CUDNN and add them to your path.
        * As of August 2017, CUDA 8 and CUDNN 5.1 (*not* 6.0) are the versions that work with TensorFlow
   4) Install TensorFlow.  (pip3 install --upgrade tensorflow)
        * We built against TensorFlow 1.2.  Earlier versions *do not* currently work.
   5) Install SciPy.  
        * On Windows, you will probably need to grab from here: http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy
		* Ensure you grab the .whl for the right Python (i.e., scipy-0.19.0-cp35-cp35m-win_amd64.whl gets SciPy 0.19 for Python 3.5)
   6) Install NumPy+MKL.
        * On Windows, you will probably need to grab from here: http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy
		* Same Python versioning issue (i.e., we grabbed numpy-1.11.3+mkl-cp35-cp35m-win_amd64.whl)
   7) Install h5py (pip3 install h5py)
        * You might be able to skip this step, as I think I removed h5py dependencies when prepping for the Falcor release.
   8) Install pillow (pip3 install pillow)
