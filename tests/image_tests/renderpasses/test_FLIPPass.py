import sys
sys.path.append('..')
from helpers import render_frames
from graphs.FLIPPass import FLIPPass as g
from falcor import *

m.addGraph(g)

# default
render_frames(m, 'default')

# useMagma
for useMagma in [False, True]:
    g.updatePass('FLIP', {'useMagma': useMagma})
    render_frames(m, 'useMagma.' + str(useMagma))

# isHDR
for isHDR in [False, True]:
    g.updatePass('FLIP', {'isHDR': isHDR})
    render_frames(m, 'isHDR.' + str(isHDR))

# toneMapper
for toneMapper in ['ACES', 'Hable', 'Reinhard']:
    g.updatePass('FLIP', {'isHDR': True, 'toneMapper': toneMapper})
    # TODO: Remove "FLIPToneMapperType." from name.
    render_frames(m, 'toneMapper.FLIPToneMapperType.' + toneMapper)

exit()
