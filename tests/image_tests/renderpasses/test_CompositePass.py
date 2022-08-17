import sys
sys.path.append('..')
from helpers import render_frames
from graphs.CompositePass import CompositePass as g
from falcor import *

m.addGraph(g)

# default
render_frames(m, 'default')

# modes
for mode in [CompositeMode.Add, CompositeMode.Multiply]:
    g.updatePass('Composite', {'mode': mode})
    render_frames(m, 'mode.' + str(mode))

# scaleA, scaleB
for scaleA, scaleB in [(0.5, 1.5), (1.0, 1.0), (1.5, 0.5)]:
    g.updatePass('Composite', {'mode': CompositeMode.Add, 'scaleA': scaleA, 'scaleB': scaleB})
    render_frames(m, 'scaleA.' + str(scaleA) + '.scaleB.' + str(scaleB))

exit()
