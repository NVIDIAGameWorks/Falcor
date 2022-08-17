import sys
sys.path.append('..')
from helpers import render_frames
from graphs.MVecRT import MVecRT as g
from falcor import *

sceneFile = 'Cerberus/Standard/Cerberus.pyscene'

m.addGraph(g)
m.loadScene(sceneFile)

# default
render_frames(m, 'default', frames=[1,16,64])

# re-load scene with 32-bit indices
m.loadScene(sceneFile, buildFlags=SceneBuilderFlags.Force32BitIndices)

render_frames(m, '32bit-indices', frames=[1,16,64])

# re-load scene with non-indexed vertices
m.loadScene(sceneFile, buildFlags=SceneBuilderFlags.NonIndexedVertices)

render_frames(m, 'non-indexed', frames=[1,16,64])

exit()
