from helpers import render_frames
from graphs.MegakernelPathTracerVBuffer import MegakernelPathTracerVBuffer as g
from falcor import *

m.addGraph(g)
m.loadScene('Arcade/Arcade.fscene')
ctx = locals()

# default
render_frames(ctx, 'default', frames=[128])

exit()
