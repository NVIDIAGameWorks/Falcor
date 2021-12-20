import sys
sys.path.append('..')
from falcor import *
from helpers import render_frames

exec(open('../../../Source/Mogwai/Data/MinimalPathTracer.py').read())

# default
render_frames(m, 'default', frames=[64])

# arcade
m.loadScene('Arcade/Arcade.pyscene')
render_frames(m, 'arcade', frames=[64])

# materials
m.loadScene('TestScenes/MaterialTest.pyscene')
render_frames(m, 'materials', frames=[64])

exit()
