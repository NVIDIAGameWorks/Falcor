import sys
sys.path.append('..')
import os
from helpers import render_frames
from graphs.SceneDebugger import SceneDebugger as SceneDebuggerGraph
from graphs.PathTracer import PathTracer as PathTracerGraph
from falcor import *

m.addGraph(PathTracerGraph)

# arcade
m.loadScene('Arcade/Arcade.pyscene', SceneBuilderFlags.RebuildCache)
render_frames(m, 'arcade', frames=[64])
m.loadScene('Arcade/Arcade.pyscene', SceneBuilderFlags.UseCache)
render_frames(m, 'arcade.cached', frames=[64])

# grey_and_white_room
m.loadScene('grey_and_white_room/grey_and_white_room.fbx', SceneBuilderFlags.RebuildCache)
render_frames(m, 'grey_and_white_room', frames=[64])
m.loadScene('grey_and_white_room/grey_and_white_room.fbx', SceneBuilderFlags.UseCache)
render_frames(m, 'grey_and_white_room.cached', frames=[64])

m.removeGraph(PathTracerGraph)
m.addGraph(SceneDebuggerGraph)

# volumes
m.loadScene(os.path.abspath('scenes/Volumes.pyscene'), SceneBuilderFlags.RebuildCache)
render_frames(m, 'volumes', frames=[1])
m.loadScene(os.path.abspath('scenes/Volumes.pyscene'), SceneBuilderFlags.UseCache)
render_frames(m, 'volumes.cached', frames=[1])

exit()
