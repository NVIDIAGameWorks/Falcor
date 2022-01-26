import sys
sys.path.append('..')
from helpers import render_frames
from graphs.PathTracerMaterials import PathTracerMaterials as g
from falcor import *

m.addGraph(g)

# Test variations of the standard material
m.loadScene('TestScenes/MaterialTest.pyscene')
render_frames(m, 'default', frames=[1,256])

# Test different material types
m.loadScene('TestScenes/Materials/Materials.pyscene')
render_frames(m, 'types', frames=[1,256])

# Test alpha testing
m.loadScene('TestScenes/AlphaTest/AlphaTest.pyscene')
render_frames(m, 'alpha', frames=[1,64])

# Test disabling alpha testing on secondary hits
g.updatePass('PathTracer', {'samplesPerPixel': 1, 'maxSurfaceBounces': 3, 'useAlphaTest': False})
render_frames(m, 'noalpha', frames=[1,64])

exit()
