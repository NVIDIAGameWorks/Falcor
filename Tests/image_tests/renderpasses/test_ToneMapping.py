import sys
sys.path.append('..')
from helpers import render_frames
from graphs.ToneMapping import ToneMapping as g
from falcor import *

m.addGraph(g)

# default
render_frames(m, 'default')

# operator
for operator in [ToneMapOp.Linear, ToneMapOp.Reinhard, ToneMapOp.ReinhardModified, ToneMapOp.HejiHableAlu, ToneMapOp.HableUc2, ToneMapOp.Aces]:
    g.updatePass('ToneMapping', {'operator': operator})
    render_frames(m, 'operator.' + str(operator))

# autoExposure
for b in [False, True]:
    g.updatePass('ToneMapping', {'autoExposure': b})
    render_frames(m, 'autoExposure.' + str(b))

# exposureCompensation
for v in [-2, 0, 2]:
    g.updatePass('ToneMapping', {'exposureCompensation': v})
    render_frames(m, 'exposureCompensation.' + str(v))

# fNumber
for v in [0.5, 1.0, 2.0]:
    g.updatePass('ToneMapping', {'autoExposure': False, 'fNumber': v})
    render_frames(m, 'fNumber.' + str(v))

# shutter
for v in [0.5, 1.0, 2.0]:
    g.updatePass('ToneMapping', {'autoExposure': False, 'shutter': v})
    render_frames(m, 'shutter.' + str(v))

# filmSpeed
for v in [50, 100, 200]:
    g.updatePass('ToneMapping', {'autoExposure': False, 'filmSpeed': v})
    render_frames(m, 'filmSpeed.' + str(v))

# whitePoint
for v in [4000, 6500, 8000]:
    g.updatePass('ToneMapping', {'whiteBalance': True, 'whitePoint': v})
    render_frames(m, 'whitePoint.' + str(v))

exit()
