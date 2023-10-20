IMAGE_TEST = {
    "platforms": ["windows-x86_64"],
    "device_types": ["d3d12", "vulkan"]
}

import sys
sys.path.append('..')
from helpers import render_frames
from graphs.PathTracerAdaptive import PathTracerAdaptive as g
from falcor import *

m.addGraph(g)
m.loadScene('Arcade/Arcade.pyscene')

# default
render_frames(m, 'default', frames=[1,16])

exit()
