import sys
sys.path.append('..')
from helpers import render_frames
from falcor import *

exec(open('../../../Source/Mogwai/Data/ForwardRenderer.py').read())
m.loadScene("TestScenes/AnimatedCubes/AnimatedCubes.pyscene")

# preInfinityBehavior (behavior before first keyframe)
render_frames(m, "preInfinity", frames=[1,90,180,270,350])

# default (behavior during defined keyframes)
render_frames(m, "default", frames=[390,410,430,450,480])

# postInfinityBehavior (behavior after last keyframe)
render_frames(m, "postInfinity", frames=[490,520,550,580])

# loopAnimations
m.scene.loopAnimations = False
m.scene.camera = m.scene.cameras[1]
render_frames(m, "loopAnimations.false", frames=[850,900,950,1000])

exit()
