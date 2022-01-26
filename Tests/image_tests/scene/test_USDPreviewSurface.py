import sys
sys.path.append('..')
import os
from helpers import render_frames
from graphs.PathTracer import PathTracer as g
from falcor import *

m.addGraph(g)

# Knob
m.loadScene('TestScenes/MoriKnob/knob.usd')
render_frames(m, 'Knob', frames=[1,64])

# KnobMaterials
m.loadScene('TestScenes/MoriKnob/KnobMaterials.usda')
render_frames(m, 'KnobMaterials', frames=[1,64])

exit()
