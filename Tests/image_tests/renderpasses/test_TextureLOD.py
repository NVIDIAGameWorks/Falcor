# use the WhittedRayTracer pass to test various texture LOD modes
import sys
sys.path.append('..')
from helpers import render_frames
from graphs.WhittedRayTracer import WhittedRayTracer as g
from falcor import *

m.addGraph(g)
m.loadScene('TestScenes/texLOD/spheres_cube.pyscene')

# default
render_frames(m, 'default')

# texLODMode
for mode in [TexLODMode.Mip0, TexLODMode.RayCones, TexLODMode.RayDiffs]:
    g.updatePass('WhittedRayTracer', {'maxBounces': 7, 'texLODMode': mode})
    render_frames(m, 'texLODMode.' + str(mode))

# rayConeFilterMode
for mode in [RayFootprintFilterMode.Isotropic, RayFootprintFilterMode.Anisotropic, RayFootprintFilterMode.AnisotropicWhenRefraction]:
    g.updatePass('WhittedRayTracer', {'maxBounces': 7, 'texLODMode': TexLODMode.RayCones, 'rayConeFilterMode': mode})
    render_frames(m, 'rayConeFilterMode.' + str(mode))

exit()
