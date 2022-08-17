import sys
sys.path.append('..')
import os
from helpers import render_frames
from graphs.TestRtProgram import TestRtProgramGraph as g
from falcor import *

m.addGraph(g)

# default: triangles and custom primitives
m.loadScene('TestScenes/GeometryTypes.pyscene')
render_frames(m, 'default', frames=[1])

# two_curves: triangles, curves, and custom primitives
m.loadScene('CurveTest/two_curves.pyscene')
render_frames(m, 'two_curves', frames=[1])

g.updatePass('TestRtProgram', {'mode': 1})

# test for dynamic dispatch
m.loadScene('TestScenes/AlphaTest/AlphaTest.pyscene')
render_frames(m, 'types', frames=[1])

exit()
