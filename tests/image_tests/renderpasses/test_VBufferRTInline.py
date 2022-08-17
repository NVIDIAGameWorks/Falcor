IMAGE_TEST = {
    'tolerance': 5e-7
}

import sys
sys.path.append('..')
from helpers import render_frames
from graphs.VBufferRTInline import VBufferRT as g
from falcor import *

m.addGraph(g)
m.loadScene('Arcade/Arcade.pyscene')

# default
render_frames(m, 'default', frames=[1,16,64])

# enable depth-of-field
m.scene.camera.focalDistance = 3.0
m.scene.camera.apertureRadius = 0.1
render_frames(m, 'dof', frames=[1,16,64])

exit()
