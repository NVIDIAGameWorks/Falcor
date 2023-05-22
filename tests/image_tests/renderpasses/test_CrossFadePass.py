IMAGE_TEST = {
    "device_types": ["d3d12", "vulkan"]
}

import sys
sys.path.append('..')
from helpers import render_frames
from graphs.CrossFadePass import CrossFadePass as g
from falcor import *

m.addGraph(g)

# default
render_frames(m, 'default', frames=[10,50])

exit()
