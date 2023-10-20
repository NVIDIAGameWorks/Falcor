IMAGE_TEST = {
    "device_types": ["d3d12", "vulkan"]
}

import sys
sys.path.append('..')
from falcor import *
from helpers import render_frames

exec(open('../../../scripts/SceneDebugger.py').read())

# default
render_frames(m, 'default', frames=[64])

# arcade
m.loadScene('Arcade/Arcade.pyscene')
render_frames(m, 'arcade', frames=[64])

exit()
