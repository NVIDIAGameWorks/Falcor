import sys
import os
sys.path.append('..')
from helpers import render_frames
from graphs.SDFEditorRenderGraphV2 import DefaultRenderGraph as g
from falcor import *

m.addGraph(g)
m.loadScene(os.path.abspath('../scene/scenes/SDFEditorSceneTwoSDFs.pyscene'))

# default
render_frames(m, 'default', frames=[64])

exit()
