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

exit()
