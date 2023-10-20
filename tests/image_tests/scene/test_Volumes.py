import sys
sys.path.append('..')
from helpers import render_frames
from graphs.SceneDebugger import SceneDebugger as g
from falcor import *

m.addGraph(g)
m.loadScene('test_scenes/two_volumes.pyscene')

# default
render_frames(m, 'default', frames=[64])

exit()
