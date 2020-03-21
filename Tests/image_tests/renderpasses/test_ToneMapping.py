from helpers import render_frames
from graphs.ToneMapping import ToneMapping as g
from falcor import *

m.addGraph(g)
ctx = locals()

# default
render_frames(ctx, 'default')

# operator
for operator in [ToneMapOp.Linear, ToneMapOp.Reinhard, ToneMapOp.ReinhardModified, ToneMapOp.HejiHableAlu, ToneMapOp.HableUc2, ToneMapOp.Aces]:
    g.updatePass('ToneMapping', {'operator': operator})
    render_frames(ctx, 'operator.' + str(operator))

# autoExposure
for b in [False, True]:
    g.updatePass('ToneMapping', {'autoExposure': b})
    render_frames(ctx, 'autoExposure.' + str(b))

# exposureCompensation
for v in [-2, 0, 2]:
    g.updatePass('ToneMapping', {'exposureCompensation': v})
    render_frames(ctx, 'exposureCompensation.' + str(v))

# exposureValue
for v in [-2, 0, 2]:
    g.updatePass('ToneMapping', {'autoExposure': False, 'exposureValue': v})
    render_frames(ctx, 'exposureValue.' + str(v))

# filmSpeed
for v in [50, 100, 200]:
    g.updatePass('ToneMapping', {'autoExposure': False, 'filmSpeed': v})
    render_frames(ctx, 'filmSpeed.' + str(v))

# whitePoint
for v in [4000, 6500, 8000]:
    g.updatePass('ToneMapping', {'whiteBalance': True, 'whitePoint': v})
    render_frames(ctx, 'whitePoint.' + str(v))

exit()
