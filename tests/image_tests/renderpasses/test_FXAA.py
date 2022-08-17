import sys
sys.path.append('..')
from helpers import render_frames
from graphs.FXAA import FXAA as g
from falcor import *

m.addGraph(g)
m.loadScene('Arcade/Arcade.pyscene')

# default
render_frames(m, 'default')

# qualitySubPix
for v in [0.25, 0.5, 0.75]:
    g.updatePass('FXAA', {'qualitySubPix': v})
    render_frames(m, 'qualitySubPix.' + str(v))

# qualityEdgeThreshold
for v in [0.166, 0.5, 0.75]:
    g.updatePass('FXAA', {'qualityEdgeThreshold': v})
    render_frames(m, 'qualityEdgeThreshold.' + str(v))

# qualityEdgeThresholdMin
for v in [0.0833, 0.25, 0.375]:
    g.updatePass('FXAA', {'qualityEdgeThresholdMin': v})
    render_frames(m, 'qualityEdgeThresholdMin.' + str(v))

# earlyOut
for b in [False, True]:
    g.updatePass('FXAA', {'earlyOut': b})
    render_frames(m, 'earlyOut.' + str(b))

exit()
