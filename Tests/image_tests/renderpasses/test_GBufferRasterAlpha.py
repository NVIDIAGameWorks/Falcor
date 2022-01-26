import sys
sys.path.append('..')
from helpers import render_frames
from graphs.GBufferRasterAlpha import GBufferRaster as g
from falcor import *

m.addGraph(g)
m.loadScene('TestScenes/AlphaTest/AlphaTest.pyscene')

# default
render_frames(m, 'default', frames=[1])

# force cull back
g.updatePass('GBufferRaster', {'forceCullMode': True, 'cull': CullMode.CullBack})
render_frames(m, 'cullback', frames=[1])

# force cull front
g.updatePass('GBufferRaster', {'forceCullMode': True, 'cull': CullMode.CullFront})
render_frames(m, 'cullfront', frames=[1])

# force cull none
g.updatePass('GBufferRaster', {'forceCullMode': True, 'cull': CullMode.CullNone})
render_frames(m, 'cullnone', frames=[1])

# disable alpha
g.updatePass('GBufferRaster', {'forceCullMode': True, 'cull': CullMode.CullNone, 'useAlphaTest': False})
render_frames(m, 'alphaoff', frames=[1])

exit()
