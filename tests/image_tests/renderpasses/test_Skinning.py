import sys
sys.path.append('..')
from helpers import render_frames
from graphs.ForwardRendering import ForwardRendering as g
from falcor import *

m.addGraph(g)
m.loadScene('Cerberus/Standard/Cerberus.pyscene')

# default
render_frames(m, 'default', frames=[1,16,64])

exit()
