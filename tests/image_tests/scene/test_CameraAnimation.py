import sys
sys.path.append('..')
from helpers import render_frames
from graphs.SceneDebugger import SceneDebugger as g
from falcor import *

m.addGraph(g)
m.loadScene("grey_and_white_room/grey_and_white_room.fbx")

# default
render_frames(m, 'default', frames=[1,16,64,128,256])

exit()
