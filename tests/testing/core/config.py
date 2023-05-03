'''
Common testing environment configuration.
'''

import os

# Default environment configuration file.
DEFAULT_ENVIRONMENT="environment/default.json"

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

# Build configurations.
BUILD_CONFIGS = {
    # Temporary build configurations combining a CMake preset and build type.
    # These should be replaced by parsing CMakePresets.json in the future.
    'windows-vs2019-Release': {
        'build_dir': 'build/windows-vs2019/bin/Release'
    },
    'windows-vs2019-Debug': {
        'build_dir': 'build/windows-vs2019/bin/Debug'
    },
    'windows-vs2019-ci-Release': {
        'build_dir': 'build/windows-vs2019-ci/bin/Release'
    },
    'windows-vs2019-ci-Debug': {
        'build_dir': 'build/windows-vs2019-ci/bin/Debug'
    },
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
    'linux-ninja-clang-Release': {
        'build_dir': 'build/linux-ninja-clang/bin/Release'
    },
    'linux-ninja-clang-Debug': {
        'build_dir': 'build/linux-ninja-clang/bin/Debug'
    },
    'linux-ninja-clang-ci-Release': {
        'build_dir': 'build/linux-ninja-clang-ci/bin/Release'
    },
    'linux-ninja-clang-ci-Debug': {
        'build_dir': 'build/linux-ninja-clang-ci/bin/Debug'
    },
}

if os.name == 'nt':
    # Executables.
    CMAKE_EXE = "tools/.packman/cmake/bin/cmake.exe"
    FALCOR_TEST_EXE = 'FalcorTest.exe'
    MOGWAI_EXE = 'Mogwai.exe'
    IMAGE_COMPARE_EXE = 'ImageCompare.exe'

    SUPPORTED_DEVICE_TYPES = ["d3d12", "vulkan"]

elif os.name == 'posix':
    # Executables.
    CMAKE_EXE = "tools/.packman/cmake/bin/cmake"
    FALCOR_TEST_EXE = 'FalcorTest'
    MOGWAI_EXE = 'Mogwai'
    IMAGE_COMPARE_EXE = 'ImageCompare'

    SUPPORTED_DEVICE_TYPES = ["vulkan"]

else:
    raise RuntimeError('Testing is not supported on this platform')
