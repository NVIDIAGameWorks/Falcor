'''
Common testing environment configuration.
'''

import os

# Default environment configuration file.
DEFAULT_ENVIRONMENT="environment/default.json"

# Image comparison tolerance.
TOLERANCE = 0.00001

IMAGE_TESTS_DIR = "Tests/image_tests"

# Supported image extensions.
IMAGE_EXTENSIONS = ['.png', '.jpg', '.tga', '.bmp', '.pfm', '.exr']

# Suffix to use for error images.
ERROR_IMAGE_SUFFIX = '.error.png'

if os.name == 'nt':

    # Build configurations.
    BUILD_CONFIGS = {
        'ReleaseD3D12': {
            'build_dir': 'Bin/x64/Release',
        },
        'DebugD3D12': {
            'build_dir': 'Bin/x64/Debug'
        },
        'ReleaseVK': {
            'build_dir': 'Bin/x64/Release'
        },
        'DebugVK': {
            'build_dir': 'Bin/x64/Debug'
        }
    }

    DEFAULT_BUILD_CONFIG = 'ReleaseD3D12'

    SOLUTION_FILE="Falcor.sln"

    # Executables.
    FALCOR_TEST_EXE = 'FalcorTest.exe'
    MOGWAI_EXE = 'Mogwai.exe'
    IMAGE_COMPARE_EXE = 'ImageCompare.exe'

else:
    raise RuntimeError('Testing is only supported on Windows')
