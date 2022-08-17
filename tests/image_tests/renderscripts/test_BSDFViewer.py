import sys
sys.path.append('..')
from falcor import *
from helpers import render_frames

exec(open('../../../Source/Mogwai/Data/BSDFViewer.py').read())

# arcade
m.loadScene('Arcade/Arcade.pyscene')
render_frames(m, 'arcade', frames=[16])

# materials
m.loadScene('TestScenes/MaterialTest.pyscene')
render_frames(m, 'materials', frames=[16])

exit()
