IMAGE_TEST = {
    'tolerance': 1e-3
}

# NOTE:
# The implementation does not seem to be deterministic.
# We're setting a larger threshold here to account for that.

import sys
sys.path.append('..')
from helpers import render_frames
from graphs.RTXGI import RTXGI as g
from falcor import *

m.addGraph(g)
m.loadScene('Arcade/Arcade.pyscene')

# default
render_frames(m, 'default', frames=[1,16,64,256])

exit()
