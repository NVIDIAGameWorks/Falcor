from helpers import render_frames
from graphs.CompositePass import CompositePass as g
from falcor import *

m.addGraph(g)
ctx = locals()

# default
render_frames(ctx, 'default')

# modes
for mode in [CompositeMode.Add, CompositeMode.Multiply]:
    g.updatePass('Composite', {'mode': mode})
    render_frames(ctx, 'mode.' + str(mode))

# scaleA, scaleB
for scaleA, scaleB in [(0.5, 1.5), (1.0, 1.0), (1.5, 0.5)]:
    g.updatePass('Composite', {'mode': CompositeMode.Add, 'scaleA': scaleA, 'scaleB': scaleB})
    render_frames(ctx, 'scaleA.' + str(scaleA) + '.scaleB.' + str(scaleB))

exit()
