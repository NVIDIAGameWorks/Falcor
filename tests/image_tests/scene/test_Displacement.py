import sys
sys.path.append('..')
import os
from helpers import render_frames
from graphs.SceneDebugger import SceneDebugger as g
from falcor import *

m.addGraph(g)
m.loadScene('test_scenes/cornell_box_displaced.pyscene')

# default
render_frames(m, 'default', frames=[64])

exit()
