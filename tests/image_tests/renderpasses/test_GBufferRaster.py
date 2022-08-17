import sys
sys.path.append('..')
from helpers import render_frames
from graphs.GBufferRaster import GBufferRaster as g
from falcor import *

m.addGraph(g)
m.loadScene('Arcade/Arcade.pyscene')

# default
render_frames(m, 'default', frames=[1,16,64])

# re-load scene with non-indexed vertices
m.loadScene('Arcade/Arcade.pyscene', buildFlags=SceneBuilderFlags.NonIndexedVertices)

render_frames(m, 'non-indexed', frames=[1,16,64])

exit()
