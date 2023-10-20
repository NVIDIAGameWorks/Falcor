'''
Common testing environment configuration.
'''

import os

# Default environment configuration file.
DEFAULT_ENVIRONMENT="tests/environment/default.json"

# Default platforms to run image tests on.
DEFAULT_PLATFORMS = ["windows-x86_64", "linux-x86_64"]

# Default image comparison tolerance.
DEFAULT_TOLERANCE = 0.0

# Default image test timeout.
DEFAULT_TIMEOUT = 600

# Default number of processes (it will be this or # of CPUs, whichever is lower)
DEFAULT_PROCESS_COUNT = 4

IMAGE_TESTS_DIR = "tests/image_tests"

# Supported image extensions.
IMAGE_EXTENSIONS = ['.png', '.jpg', '.tga', '.bmp', '.pfm', '.exr']

# Suffix to use for error images.
ERROR_IMAGE_SUFFIX = '.error.png'

PYTHON_TESTS_DIR = "tests/python_tests"

# Build configurations.
BUILD_CONFIGS = {
    # Temporary build configurations combining a CMake preset and build type.
    # These should be replaced by parsing CMakePresets.json in the future.
    'windows-vs2022-Release': {
        'build_dir': 'build/windows-vs2022/bin/Release'
    },
    'windows-vs2022-Debug': {
        'build_dir': 'build/windows-vs2022/bin/Debug'
    },
    'windows-vs2022-ci-Release': {
        'build_dir': 'build/windows-vs2022-ci/bin/Release'
    },
    'windows-vs2022-ci-Debug': {
        'build_dir': 'build/windows-vs2022-ci/bin/Debug'
    },
    'windows-ninja-msvc-Release': {
        'build_dir': 'build/windows-ninja-msvc/bin/Release'
    },
    'windows-ninja-msvc-Debug': {
        'build_dir': 'build/windows-ninja-msvc/bin/Debug'
    },
    'windows-ninja-msvc-ci-Release': {
        'build_dir': 'build/windows-ninja-msvc-ci/bin/Release'
    },
    'windows-ninja-msvc-ci-Debug': {
        'build_dir': 'build/windows-ninja-msvc-ci/bin/Debug'
    },
    'linux-clang-Release': {
        'build_dir': 'build/linux-clang/bin/Release'
    },
    'linux-clang-Debug': {
        'build_dir': 'build/linux-clang/bin/Debug'
    },
    'linux-clang-ci-Release': {
        'build_dir': 'build/linux-clang-ci/bin/Release'
    },
    'linux-clang-ci-Debug': {
        'build_dir': 'build/linux-clang-ci/bin/Debug'
    },
    'linux-gcc-Release': {
        'build_dir': 'build/linux-gcc/bin/Release'
    },
    'linux-gcc-Debug': {
        'build_dir': 'build/linux-gcc/bin/Debug'
    },
    'linux-gcc-ci-Release': {
        'build_dir': 'build/linux-gcc-ci/bin/Release'
    },
    'linux-gcc-ci-Debug': {
        'build_dir': 'build/linux-gcc-ci/bin/Debug'
    },
}

if os.name == 'nt':
    PLATFORM = "windows-x86_64"
    # Executables.
    CMAKE_EXE = "tools/.packman/cmake/bin/cmake.exe"
    FALCOR_LIB = 'Falcor.dll'
    FALCOR_TEST_EXE = 'FalcorTest.exe'
    MOGWAI_EXE = 'Mogwai.exe'
    IMAGE_COMPARE_EXE = 'ImageCompare.exe'
    PYTHON_EXE = "pythondist/python.exe"

    SUPPORTED_DEVICE_TYPES = ["d3d12", "vulkan"]

elif os.name == 'posix':
    PLATFORM = "linux-x86_64"
    # Executables.
    CMAKE_EXE = "tools/.packman/cmake/bin/cmake"
    FALCOR_LIB = 'libFalcor.so'
    FALCOR_TEST_EXE = 'FalcorTest'
    MOGWAI_EXE = 'Mogwai'
    IMAGE_COMPARE_EXE = 'ImageCompare'
    PYTHON_EXE = "pythondist/python"

    SUPPORTED_DEVICE_TYPES = ["vulkan"]

else:
    raise RuntimeError('Testing is not supported on this platform')
