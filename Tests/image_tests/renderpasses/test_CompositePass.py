from helpers import render_frames
from graphs.CompositePass import CompositePass as g
from falcor import *

m.addGraph(g)
ctx = locals()

# default
render_frames(ctx, 'default')

# scaleA, scaleB
for scaleA, scaleB in [(0.5, 1.5), (1.0, 1.0), (1.5, 0.5)]:
    g.updatePass('Composite', {'scaleA': scaleA, 'scaleB': scaleB})
    render_frames(ctx, 'scaleA.' + str(scaleA) + '.scaleB.' + str(scaleB))

exit()
