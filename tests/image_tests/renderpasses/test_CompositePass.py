IMAGE_TEST = {
    "device_types": ["d3d12", "vulkan"]
}

import sys
sys.path.append('..')
from helpers import render_frames
from graphs.CompositePass import CompositePass as g
from falcor import *

m.addGraph(g)

# default
render_frames(m, 'default')

# modes
for mode in ['Add', 'Multiply']:
    g.updatePass('Composite', {'mode': mode})
    # TODO: Remove "CompositeMode." from name.
    render_frames(m, 'mode.CompositeMode.' + mode)

# scaleA, scaleB
for scaleA, scaleB in [(0.5, 1.5), (1.0, 1.0), (1.5, 0.5)]:
    g.updatePass('Composite', {'mode': 'Add', 'scaleA': scaleA, 'scaleB': scaleB})
    render_frames(m, 'scaleA.' + str(scaleA) + '.scaleB.' + str(scaleB))

exit()
