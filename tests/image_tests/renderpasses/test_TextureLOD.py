# use the WhittedRayTracer pass to test various texture LOD modes
import sys
sys.path.append('..')
from helpers import render_frames
from graphs.WhittedRayTracer import WhittedRayTracer as g
from falcor import *

m.addGraph(g)
m.loadScene('test_scenes/tex_lod/spheres_cube.pyscene')

# default
render_frames(m, 'default')

# texLODMode
for mode in ['Mip0', 'RayCones', 'RayDiffs']:
    g.updatePass('WhittedRayTracer', {'maxBounces': 7, 'texLODMode': mode})
    # TODO: Remove "TexLODMode." from name.
    render_frames(m, 'texLODMode.TexLODMode.' + mode)

# rayConeFilterMode
for mode in ['Isotropic', 'Anisotropic', 'AnisotropicWhenRefraction']:
    g.updatePass('WhittedRayTracer', {'maxBounces': 7, 'texLODMode': 'RayCones', 'rayConeFilterMode': mode})
    # TODO: Remove "RayFootprintFilterMode." from name.
    render_frames(m, 'rayConeFilterMode.RayFootprintFilterMode.' + mode)

exit()
