import sys
sys.path.append('..')
import os
from helpers import render_frames
from graphs.SceneDebugger import SceneDebugger as SceneDebuggerGraph
from graphs.GBufferRTCullBack import GBufferRTCullBack
from graphs.GBufferRasterCullBack import GBufferRasterCullBack
from falcor import *

# Load test scene that has mixed triangle winding in object/world space
m.loadScene('TestScenes/WindingTest.pyscene')

m.addGraph(SceneDebuggerGraph)
SceneDebuggerGraph.getPass('SceneDebugger').mode = SceneDebuggerMode.FrontFacingFlag

render_frames(m, 'frontfacing', frames=[2])

SceneDebuggerGraph.getPass('SceneDebugger').mode = SceneDebuggerMode.FaceNormal

render_frames(m, 'facenormal', frames=[2])

m.removeGraph(SceneDebuggerGraph)
m.addGraph(GBufferRTCullBack)

render_frames(m, 'rt_cullback', frames=[2])

m.removeGraph(GBufferRTCullBack)
m.addGraph(GBufferRasterCullBack)

render_frames(m, 'raster_cullback', frames=[2])

exit()
