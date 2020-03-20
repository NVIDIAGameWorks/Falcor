from helpers import render_frames
from graphs.SplitScreen import SplitScreen as g
from falcor import *

m.addGraph(g)
ctx = locals()

# default
render_frames(ctx, 'default')

# splitLocation
for v in [0.5, 0.25, 0.75]:
    g.updatePass('SplitScreenPass', {'splitLocation': v})
    render_frames(ctx, 'splitLocation.' + str(v))

exit()
