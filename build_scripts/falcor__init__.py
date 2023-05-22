import sys as _sys
import os as _os

if _os.name == "nt" and _sys.version_info >= (3, 8):
    falcor_dir = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "../.."))
    _os.add_dll_directory(falcor_dir)
    del falcor_dir

del _sys, _os

from .falcor_ext import *
