import sys
sys.path.append('..')
from helpers import render_frames

exec(open('../../../Source/Mogwai/Data/RTXGI.py').read())

# default
render_frames(m, 'default', frames=[64])

# arcade
m.loadScene('Arcade/Arcade.pyscene')
# We do not capture any frames because due to probe relocation the result image is not deterministic.
# We still want to load the scene to make sure the script works.
# render_frames(m, 'arcade', frames=[256])

exit()
