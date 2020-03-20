from helpers import render_frames
from graphs.FXAA import FXAA as g
from falcor import *

m.addGraph(g)
m.loadScene('Arcade/Arcade.fscene')
ctx = locals()

# default
render_frames(ctx, 'default', frames=[1,16,64])

# qualitySubPix
for v in [0.25, 0.5, 0.75]:
    g.updatePass('FXAA', {'qualitySubPix': v})
    render_frames(ctx, 'qualitySubPix.' + str(v), frames=[1,16,64])

# qualityEdgeThreshold
for v in [0.166, 0.5, 0.75]:
    g.updatePass('FXAA', {'qualityEdgeThreshold': v})
    render_frames(ctx, 'qualityEdgeThreshold.' + str(v), frames=[1,16,64])

# qualityEdgeThresholdMin
for v in [0.0833, 0.25, 0.375]:
    g.updatePass('FXAA', {'qualityEdgeThresholdMin': v})
    render_frames(ctx, 'qualityEdgeThresholdMin.' + str(v), frames=[1,16,64])

# earlyOut
for b in [False, True]:
    g.updatePass('FXAA', {'earlyOut': b})
    render_frames(ctx, 'earlyOut.' + str(b), frames=[1,16,64])

exit()
