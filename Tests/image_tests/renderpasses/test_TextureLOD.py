# use the WhittedRayTracer pass to test various texture LOD modes
import sys
sys.path.append('..')
from helpers import render_frames
from graphs.WhittedRayTracer import WhittedRayTracer as g
from falcor import *

m.addGraph(g)
m.loadScene('Arcade/Arcade.pyscene')

# default
render_frames(m, 'default')

# texture LOD mode
for mode in [TextureLODMode.Mip0, TextureLODMode.RayCones, TextureLODMode.RayDiffsIsotropic, TextureLODMode.RayDiffsAnisotropic]:
    g.updatePass('WhittedRayTracer', {'mTexLODMode': mode, 'mUsingRasterizedGBuffer': True, 'mMaxBounces': 1})
    render_frames(m, 'mode.' + str(mode))
exit()
