import sys
sys.path.append('..')
from helpers import render_frames

exec(open('../../../scripts/RTXDI.py').read())

# default
render_frames(m, 'default', frames=[64])

# arcade
m.loadScene('Arcade/Arcade.pyscene')
render_frames(m, 'arcade', frames=[64])

exit()
