import sys
sys.path.append('..')
from helpers import render_frames
from graphs.GaussianBlur import GaussianBlur as g
from falcor import *

m.addGraph(g)
m.loadScene("Arcade/Arcade.pyscene")

# default
render_frames(m, 'default')

# kernelWidth, sigma
for kernelWidth, sigma in [(5, 1), (9, 1.5), (15, 2)]:
    g.updatePass('GaussianBlur', {'kernelWidth': kernelWidth, 'sigma': sigma})
    render_frames(m, 'kernelWidth.' + str(kernelWidth) + '.sigma.' + str(sigma))

exit()
