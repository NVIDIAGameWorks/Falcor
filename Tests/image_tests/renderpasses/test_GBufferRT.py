from helpers import render_frames
from graphs.GBufferRT import GBufferRT as g
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

# re-load scene with non-indexed vertices
m.loadScene('Arcade/Arcade.fscene', buildFlags=SceneBuilderFlags.NonIndexedVertices)
render_frames(ctx, 'non-indexed', frames=[1,16,64])

exit()
