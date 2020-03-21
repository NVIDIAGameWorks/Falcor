from helpers import render_frames
from graphs.SideBySide import SideBySide as g
from falcor import *

m.addGraph(g)
ctx = locals()

# default
render_frames(ctx, 'default')

# imageLeftBound
for v in [250, 500, 750]:
	g.updatePass('SideBySidePass', {'imageLeftBound': v})
	render_frames(ctx, 'imageLeftBound.' + str(v))

exit()
