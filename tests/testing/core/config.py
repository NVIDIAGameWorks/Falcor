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

# Path to the bundled cmake
CMAKE_EXE = "tools/.packman/cmake/bin/cmake.exe"

if os.name == 'nt':

    # Build configurations.
    BUILD_CONFIGS = {
        # Temporary build configurations combining a CMake preset and build type.
        # These should be replaced by parsing CMakePresets.json in the future.
        'windows-vs2019-d3d12-Release': {
            'build_dir': 'build/windows-vs2019-d3d12/bin/Release'
        },
        'windows-vs2019-d3d12-Debug': {
            'build_dir': 'build/windows-vs2019-d3d12/bin/Debug'
        },
        'windows-vs2019-gfx-d3d12-Release': {
            'build_dir': 'build/windows-vs2019-gfx-d3d12/bin/Release'
        },
        'windows-vs2022-d3d12-Release': {
            'build_dir': 'build/windows-vs2022-d3d12/bin/Release'
        },
        'windows-vs2022-d3d12-Debug': {
            'build_dir': 'build/windows-vs2022-d3d12/bin/Debug'
        },
        'windows-vs2022-gfx-d3d12-Release': {
            'build_dir': 'build/windows-vs2022-gfx-d3d12/bin/Release'
        },
        'windows-ninja-msvc-d3d12-Release': {
            'build_dir': 'build/windows-ninja-msvc-d3d12/bin/Release'
        },
        'windows-ninja-msvc-d3d12-Debug': {
            'build_dir': 'build/windows-ninja-msvc-d3d12/bin/Debug'
        },
        'windows-ninja-msvc-gfx-d3d12-Release': {
            'build_dir': 'build/windows-ninja-msvc-gfx-d3d12/bin/Release'
        },
        'windows-ninja-msvc-gfx-d3d12-Debug': {
            'build_dir': 'build/windows-ninja-msvc-gfx-d3d12/bin/Debug'
        },
        'windows-ninja-msvc-gfx-vk-Release': {
            'build_dir': 'build/windows-ninja-msvc-gfx-vk/bin/Release'
        },
        'windows-ninja-msvc-gfx-vk-Debug': {
            'build_dir': 'build/windows-ninja-msvc-gfx-vk/bin/Debug'
        },
    }

    SOLUTION_FILE="Falcor.sln"

    # Executables.
    FALCOR_TEST_EXE = 'FalcorTest.exe'
    MOGWAI_EXE = 'Mogwai.exe'
    IMAGE_COMPARE_EXE = 'ImageCompare.exe'

else:
    raise RuntimeError('Testing is only supported on Windows')
