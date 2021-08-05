import sys
sys.path.append('..')
import os
from helpers import render_frames
from graphs.SceneDebugger import SceneDebugger as g
from falcor import *

m.addGraph(g)
m.loadScene('TestScenes/CornellBoxDisplaced.pyscene')

# default
render_frames(m, 'default', frames=[64])

exit()
