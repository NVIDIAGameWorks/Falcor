import sys
sys.path.append('..')
import os
from helpers import render_frames
from graphs.TestRtProgram import TestRtProgramGraph as g
from falcor import *

m.addGraph(g)

# default: triangles and custom primitives
m.loadScene('test_scenes/geometry_types.pyscene')
render_frames(m, 'default', frames=[1])

# two_curves: triangles, curves, and custom primitives
m.loadScene('test_scenes/curves/two_curves.pyscene')
render_frames(m, 'two_curves', frames=[1])

g.updatePass('TestRtProgram', {'mode': 1})

# test for dynamic dispatch
m.loadScene('test_scenes/alpha_test/alpha_test.pyscene')
render_frames(m, 'types', frames=[1])

exit()
