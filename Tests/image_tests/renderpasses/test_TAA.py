from helpers import render_frames
from graphs.TAA import TAA as g
from falcor import *

m.addGraph(g)
m.loadScene('Arcade/Arcade.fscene')
ctx = locals()

# default
render_frames(ctx, 'default', frames=[1,16,64])

# alpha
for v in [0.0, 0.5, 1.0]:
    g.updatePass('TAA', {'alpha': v})
    render_frames(ctx, 'alpha.' + str(v), frames=[1,16,64])

# colorBoxSigma
for v in [0.0, 7.5, 15.0]:
    g.updatePass('TAA', {'colorBoxSigma': v})
    render_frames(ctx, 'colorBoxSigma.' + str(v), frames=[1,16,64])

exit()
