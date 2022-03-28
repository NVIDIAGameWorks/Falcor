import sys
sys.path.append('..')
from helpers import render_frames
from graphs.PathTracer import PathTracer as g
from falcor import *

m.addGraph(g)

# default
m.loadScene('TestScenes/CornellBox.pyscene')
render_frames(m, 'default', frames=[1])

# load other scene
m.loadScene('TestScenes/NestedDielectrics.pyscene')
render_frames(m, 'nested', frames=[1])

exit()
