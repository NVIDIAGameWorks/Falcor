IMAGE_TEST = {
    'tolerance': 1e-6
}

import sys
sys.path.append('..')
from helpers import render_frames
from graphs.MegakernelPathTracerVBuffer import MegakernelPathTracerVBuffer as g
from falcor import *

m.addGraph(g)
m.loadScene('Arcade/Arcade.pyscene')

# default
render_frames(m, 'default', frames=[128])

exit()
