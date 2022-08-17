# Test different texture gradient modes in GBufferRT
import sys
sys.path.append('..')
from helpers import render_frames
from graphs.GBufferRTTexGrads import GBufferRT as g
from falcor import *

m.addGraph(g)
m.loadScene('TestScenes/texLOD/spheres_cube.pyscene')

texLODModes = [TexLODMode.Mip0, TexLODMode.RayCones, TexLODMode.RayDiffs]

# texGrads
for mode in texLODModes:
    g.updatePass('GBufferRT', {'texLOD': mode, "useTraceRayInline": False})
    render_frames(m, 'texGrads.' + str(mode))

# texGrads trace ray inline
for mode in texLODModes:
    g.updatePass('GBufferRT', {'texLOD': mode, "useTraceRayInline": True})
    render_frames(m, 'texGrads-inline.' + str(mode))

exit()
