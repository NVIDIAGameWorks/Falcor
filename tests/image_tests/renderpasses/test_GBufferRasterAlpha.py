import sys
sys.path.append('..')
from helpers import render_frames
from graphs.GBufferRasterAlpha import GBufferRaster as g
from falcor import *

m.addGraph(g)
m.loadScene('test_scenes/alpha_test/alpha_test.pyscene')

# default
render_frames(m, 'default', frames=[1])

# force cull back
g.updatePass('GBufferRaster', {'forceCullMode': True, 'cull': 'Back'})
render_frames(m, 'cullback', frames=[1])

# force cull front
g.updatePass('GBufferRaster', {'forceCullMode': True, 'cull': 'Front'})
render_frames(m, 'cullfront', frames=[1])

# force cull none
g.updatePass('GBufferRaster', {'forceCullMode': True, 'cull': 'None'})
render_frames(m, 'cullnone', frames=[1])

# disable alpha
g.updatePass('GBufferRaster', {'forceCullMode': True, 'cull': 'None', 'useAlphaTest': False})
render_frames(m, 'alphaoff', frames=[1])

exit()
