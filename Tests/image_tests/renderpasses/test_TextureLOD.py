# use the WhittedRayTracer pass to test various texture LOD modes
from helpers import render_frames
from graphs.WhittedRayTracer import WhittedRayTracer as g
from falcor import *

m.addGraph(g)
m.loadScene('Arcade/Arcade.fscene')
ctx = locals()

# default
render_frames(ctx, 'default')

# texture LOD mode
for mode in [TextureLODMode.Mip0, TextureLODMode.RayCones, TextureLODMode.RayDiffsIsotropic, TextureLODMode.RayDiffsAnisotropic]:
    g.updatePass('WhittedRayTracer', {'mTexLODMode': mode, 'mUsingRasterizedGBuffer': True, 'mMaxBounces': 1})
    render_frames(ctx, 'mode.' + str(mode))
exit()
