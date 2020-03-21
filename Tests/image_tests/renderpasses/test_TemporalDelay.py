from helpers import render_frames
from graphs.TemporalDelay import TemporalDelay as g
from falcor import *

m.addGraph(g)
m.loadScene('Arcade/Arcade.fscene')
ctx = locals()

# default
render_frames(ctx, 'default', frames=[16,17])

# delay
g.updatePass('TemporalDelayPass', {'delay': 0})
render_frames(ctx, 'delay.' + str(1))

g.updatePass('TemporalDelayPass', {'delay': 32})
render_frames(ctx, 'delay.' + str(32), frames=[32,33])

exit()
