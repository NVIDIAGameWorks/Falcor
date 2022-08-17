import sys
sys.path.append('..')
from helpers import render_frames
from graphs.ColorMapPass import ColorMapPass as g
from falcor import *

m.addGraph(g)

# default
render_frames(m, 'default')

# colorMap
for colorMap in [ColorMap.Grey, ColorMap.Jet, ColorMap.Viridis, ColorMap.Plasma, ColorMap.Magma, ColorMap.Inferno]:
    g.updatePass('ColorMap', {'colorMap': colorMap})
    render_frames(m, 'colorMap.' + str(colorMap))

# channel
for channel in [0, 1, 2, 3]:
    g.updatePass('ColorMap', {'channel': channel})
    render_frames(m, 'colorMap.channel.' + str(channel))

# minValue, maxValue
for (minValue, maxValue) in [(0, 1), (1, 0), (0.25, 0.75)]:
    g.updatePass('ColorMap', {'colorMap': ColorMap.Jet, 'minValue': minValue, 'maxValue': maxValue})
    render_frames(m, 'minValue.' + str(minValue) + '.maxValue.' + str(maxValue))

# autoRange
g.updatePass('ColorMap', {'autoRange': True, 'minValue': 0.4, 'maxValue': 0.6})
render_frames(m, 'colorMap.autoRange', frames=[1,2])

exit()
