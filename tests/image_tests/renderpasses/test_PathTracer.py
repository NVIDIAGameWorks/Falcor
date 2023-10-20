IMAGE_TEST = {
    "device_types": ["d3d12", "vulkan"]
}

import sys
sys.path.append('..')
from helpers import render_frames
from graphs.PathTracer import PathTracer as g
from falcor import *

m.addGraph(g)
m.loadScene('Arcade/Arcade.pyscene')

# default
g["PathTracer"].set_properties({"useSER": False})
render_frames(m, 'default', frames=[128])

exit()
