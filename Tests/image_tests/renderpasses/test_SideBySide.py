import sys
sys.path.append('..')
from helpers import render_frames
from graphs.SideBySide import SideBySide as g
from falcor import *

m.addGraph(g)

# default
render_frames(m, 'default')

# imageLeftBound
for v in [250, 500, 750]:
	g.updatePass('SideBySidePass', {'imageLeftBound': v})
	render_frames(m, 'imageLeftBound.' + str(v))

exit()
