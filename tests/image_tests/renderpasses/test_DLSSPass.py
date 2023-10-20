IMAGE_TEST = {
    # The test uses GBufferRaster which is not supported on Vulkan.
    "device_types": ["d3d12"],
    'tolerance': 1e-7
}

# NOTE:
# DLSS seems to be non-deterministic in some cases even with identical inputs.
# We're setting a larger threshold here to account for that.

import sys
sys.path.append('..')
from helpers import render_frames
from graphs.DLSS import DLSS as g
from falcor import *

m.addGraph(g)
m.loadScene('test_scenes/cesium_man/CesiumMan.pyscene')

# default
render_frames(m, 'default', frames=[64, 128, 192, 256])

exit()
