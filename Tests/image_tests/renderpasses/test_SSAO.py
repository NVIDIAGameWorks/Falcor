import sys
sys.path.append('..')
from helpers import render_frames
from graphs.SSAO import SSAO as g
from falcor import *

m.addGraph(g)
m.loadScene('Arcade/Arcade.pyscene')

# default
render_frames(m, 'default')

# radius
for v in [0.5, 1.0, 5.0]:
    g.updatePass('SSAO', {'radius': v})
    render_frames(m, 'radius.' + str(v))

# kernelSize
for v in [1, 5, 10]:
    g.updatePass('SSAO', {'kernelSize': v})
    render_frames(m, 'kernelSize.' + str(v))

# distribution
for distribution in [SampleDistribution.Random, SampleDistribution.UniformHammersley, SampleDistribution.CosineHammersley]:
    g.updatePass('SSAO', {'distribution': distribution})
    render_frames(m, 'distribution.' + str(distribution))

exit()
