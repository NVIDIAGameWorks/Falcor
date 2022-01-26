import sys
sys.path.append('..')
from helpers import render_frames
from graphs.ModulateIllumination import ModulateIllumination as g
from falcor import *

m.addGraph(g)

# default
render_frames(m, 'default')

exit()
