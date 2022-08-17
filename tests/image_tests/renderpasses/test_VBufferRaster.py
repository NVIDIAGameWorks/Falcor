import sys
sys.path.append('..')
from helpers import render_frames
from graphs.VBufferRaster import VBufferRaster as g
from falcor import *

m.addGraph(g)
m.loadScene('Arcade/Arcade.pyscene')

# default
render_frames(m, 'default', frames=[1,16,64])

exit()
