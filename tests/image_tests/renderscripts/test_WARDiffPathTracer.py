IMAGE_TEST = {
    'skipped': 'Skipped due to incorrect codegen when used with Slang 2025.5.3.',
    'tolerance': 1e-8
}

import sys
sys.path.append('..')
from falcor import *
from helpers import render_frames

exec(open('../../../scripts/WARDiffPathTracer.py').read())

# default
render_frames(m, 'default', frames=[64])

exit()
