lz4_stream - A C++ stream using LZ4 (de)compression
===================================================

lz4_stream is a simple wrapper that uses C++ streams for compressing and decompressing data using the [LZ4 compression library]

Usage
-----

Look at lz4\_compress.cpp and lz4\_decompress.cpp for example command line programs that can compress and decompress using this stream library.

Building
--------

```
mkdir build
cd build
cmake ..
make
```

Requirements
------------

The [LZ4 compression library] is required to use this library.

Build status
------------

Ubuntu and OSX (GCC/Clang):

[![Build Status](https://travis-ci.org/laudrup/lz4_stream.png)](https://travis-ci.org/laudrup/lz4_stream)

Windows (MS C++):

[![Build status](https://ci.appveyor.com/api/projects/status/xrp8bjf9217broom?svg=true)](https://ci.appveyor.com/project/laudrup/lz4-stream)

Code coverage (codecov.io):

[![codecov](https://codecov.io/gh/laudrup/lz4_stream/branch/master/graph/badge.svg)](https://codecov.io/gh/laudrup/lz4_stream)

License
-------

Standard BSD 3-Clause License as used by the LZ4 library.

[LZ4 compression library]: https://github.com/lz4/lz4
[cmake]: http://cmake.org
[Google Test Framework]: https://github.com/google/googletest
