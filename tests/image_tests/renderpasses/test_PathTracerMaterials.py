IMAGE_TEST = {
    "device_types": ["d3d12", "vulkan"]
}

import sys

sys.path.append("..")
from helpers import render_frames
from graphs.PathTracerMaterials import PathTracerMaterials as g
from falcor import *

m.addGraph(g)

# Test variations of the standard material
m.loadScene("test_scenes/material_test.pyscene")
render_frames(m, "default", frames=[1, 256])

# Test different material types
m.loadScene("test_scenes/materials/materials.pyscene")
render_frames(m, "types", frames=[1, 256])

# Test for light leaks
m.loadScene("test_scenes/materials/light_leaks.pyscene")
render_frames(m, "leaks", frames=[1, 256])

# Test alpha testing
m.loadScene("test_scenes/alpha_test/alpha_test.pyscene")
render_frames(m, "alpha", frames=[1, 64])

# Test disabling alpha testing on secondary hits
g["PathTracer"].reset()
g["PathTracer"].set_properties({"useAlphaTest": False})
render_frames(m, "noalpha", frames=[1, 64])

exit()
