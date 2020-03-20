from helpers import render_frames
from graphs.BSDFViewer import BSDFViewer as g
from falcor import *

m.addGraph(g)
m.loadScene('Arcade/Arcade.fscene')
ctx = locals()

# default
render_frames(ctx, 'default', frames=[64])

exit()
