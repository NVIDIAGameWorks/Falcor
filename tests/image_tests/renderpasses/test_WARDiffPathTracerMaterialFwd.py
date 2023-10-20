IMAGE_TEST = {
    'tolerance': 1e-8
}

import sys
sys.path.append('..')
from helpers import render_frames
from graphs.WARDiffPathTracerMaterialFwd import WARDiffPathTracer as g
from falcor import *

m.addGraph(g)
flags = SceneBuilderFlags.DontMergeMaterials | SceneBuilderFlags.RTDontMergeDynamic | SceneBuilderFlags.DontOptimizeMaterials
m.loadScene("test_scenes/bunny_war_diff_pt.pyscene", buildFlags=flags)

# default
render_frames(m, 'default', frames=[1024], resolution=[512, 512])

exit()
