from helpers import render_frames
from graphs.SVGF import SVGF as g
from falcor import *

m.addGraph(g)
m.loadScene('Arcade/Arcade.fscene')
ctx = locals()

# default
render_frames(ctx, 'default', frames=[1,16,64])

exit()
