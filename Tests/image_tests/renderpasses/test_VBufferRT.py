from helpers import render_frames
from graphs.VBufferRT import VBufferRT as g
from falcor import *

m.addGraph(g)
m.loadScene('Arcade/Arcade.fscene')
ctx = locals()

# default
render_frames(ctx, 'default', frames=[1,16,64])

# enable depth-of-field
m.scene.camera.focalDistance = 3.0
m.scene.camera.apertureRadius = 0.1
render_frames(ctx, 'dof', frames=[1,16,64])

exit()
