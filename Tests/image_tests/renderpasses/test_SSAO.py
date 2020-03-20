from helpers import render_frames
from graphs.SSAO import SSAO as g
from falcor import *

m.addGraph(g)
m.loadScene('Arcade/Arcade.fscene')
ctx = locals()

# default
render_frames(ctx, 'default', frames=[1,16,64])

# radius
for v in [0.5, 1.0, 5.0]:
    g.updatePass('SSAO', {'radius': v})
    render_frames(ctx, 'radius.' + str(v), frames=[1,16,64])

# kernelSize
for v in [1, 5, 10]:
    g.updatePass('SSAO', {'kernelSize': v})
    render_frames(ctx, 'kernelSize.' + str(v), frames=[1,16,64])

# distribution
for distribution in [SampleDistribution.Random, SampleDistribution.UniformHammersley, SampleDistribution.CosineHammersley]:
    g.updatePass('SSAO', {'distribution': distribution})
    render_frames(ctx, 'distribution.' + str(distribution), frames=[1,16,64])

exit()
