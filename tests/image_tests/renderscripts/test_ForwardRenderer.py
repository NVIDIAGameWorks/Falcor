import sys
sys.path.append('..')
from falcor import *
from helpers import render_frames

exec(open('../../../Source/Mogwai/Data/ForwardRenderer.py').read())

# default
render_frames(m, 'default', frames=[64])

# arcade
m.loadScene('Arcade/Arcade.pyscene')
render_frames(m, 'arcade', frames=[64])

exit()
