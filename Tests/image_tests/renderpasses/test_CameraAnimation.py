from helpers import render_frames
from graphs.ForwardRendering import ForwardRendering as g
from falcor import *

g.unmarkOutput("ForwardLightingPass.motionVecs")
m.addGraph(g)
m.loadScene("grey_and_white_room/grey_and_white_room.fbx")
ctx = locals()

# default
render_frames(ctx, 'default', frames=[1,16,64,128,256])

exit()
