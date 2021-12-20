IMAGE_TEST = {
    'tolerance': 1e-10
}

import sys
sys.path.append('..')
from helpers import render_frames
from graphs.PathTracerMaterials import PathTracerMaterials as g
from falcor import *

m.addGraph(g)
m.loadScene('TestScenes/MaterialTest.pyscene')

# default
render_frames(m, 'default', frames=[1,256])

exit()
