IMAGE_TEST = {
    'tolerance': 1e-9
}

import sys
sys.path.append('..')
from helpers import render_frames
from graphs.GBufferRTInline import GBufferRT as g
from falcor import *

m.addGraph(g)
m.loadScene('Arcade/Arcade.pyscene')

# default
render_frames(m, 'default', frames=[1,16,64])

# enable depth-of-field
m.scene.camera.focalDistance = 3.0
m.scene.camera.apertureRadius = 0.1
render_frames(m, 'dof', frames=[1,16,64])

# re-load scene with non-indexed vertices
m.loadScene('Arcade/Arcade.pyscene', buildFlags=SceneBuilderFlags.NonIndexedVertices)
render_frames(m, 'non-indexed', frames=[1,16,64])

exit()
