import sys
sys.path.append('..')
from falcor import *
from helpers import render_frames

exec(open('../../../scripts/BSDFViewer.py').read())

# arcade
m.loadScene('Arcade/Arcade.pyscene')
render_frames(m, 'arcade', frames=[16])

# materials
m.loadScene('test_scenes/material_test.pyscene')
render_frames(m, 'materials', frames=[16])

exit()
