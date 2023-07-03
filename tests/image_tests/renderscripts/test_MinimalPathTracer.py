import sys
sys.path.append('..')
from falcor import *
from helpers import render_frames

exec(open('../../../scripts/MinimalPathTracer.py').read())

# default
render_frames(m, 'default', frames=[64])

# arcade
m.loadScene('Arcade/Arcade.pyscene')
render_frames(m, 'arcade', frames=[64])

# materials
m.loadScene('test_scenes/material_test.pyscene')
render_frames(m, 'materials', frames=[64])

exit()
