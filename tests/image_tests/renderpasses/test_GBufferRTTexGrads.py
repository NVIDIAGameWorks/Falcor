# Test different texture gradient modes in GBufferRT
import sys
sys.path.append('..')
from helpers import render_frames
from graphs.GBufferRTTexGrads import GBufferRT as g
from falcor import *

m.addGraph(g)
m.loadScene('test_scenes/tex_lod/spheres_cube.pyscene')

texLODModes = ['Mip0', 'RayCones', 'RayDiffs']

# texGrads
for mode in texLODModes:
    g.updatePass('GBufferRT', {'texLOD': mode, "useTraceRayInline": False})
    # TODO: Remove "TexLODMode." from name.
    render_frames(m, 'texGrads.TexLODMode.' + mode)

# texGrads trace ray inline
for mode in texLODModes:
    g.updatePass('GBufferRT', {'texLOD': mode, "useTraceRayInline": True})
    # TODO: Remove "TexLODMode." from name.
    render_frames(m, 'texGrads-inline.TexLODMode.' + mode)

exit()
