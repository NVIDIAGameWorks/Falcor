from helpers import render_frames
from graphs.GBufferRaster import GBufferRaster as g
from falcor import *

m.addGraph(g)
m.loadScene('Arcade/Arcade.fscene')
ctx = locals()

# default
render_frames(ctx, 'default', frames=[1,16,64])

# re-load scene with non-indexed vertices
m.loadScene('Arcade/Arcade.fscene', buildFlags=SceneBuilderFlags.NonIndexedVertices)

render_frames(ctx, 'non-indexed', frames=[1,16,64])

exit()
