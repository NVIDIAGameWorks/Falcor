IMAGE_TEST = {
    "device_types": ["d3d12", "vulkan"]
}

import sys
sys.path.append('..')
from helpers import render_frames
from graphs.SceneDebugger import SceneDebugger as g
from falcor import *

m.addGraph(g)
m.loadScene('test_scenes/cesium_man/CesiumMan.pyscene')

# default
render_frames(m, 'default', frames=[1,16,64])

exit()
