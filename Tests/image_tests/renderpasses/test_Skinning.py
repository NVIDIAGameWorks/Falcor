from helpers import render_frames
from graphs.ForwardRendering import ForwardRendering as g
from falcor import *

m.addGraph(g)
m.loadScene('Cerberus/Standard/Cerberus.fscene')
ctx = locals()

# default
render_frames(ctx, 'default', frames=[1,16,64])

exit()
