IMAGE_TEST = {
    'skipped': 'Skipped due to instability on testing agents.'
}

import sys
sys.path.append('..')
from helpers import render_frames

exec(open('../../../Source/Mogwai/Data/DenoisedPathTracer.py').read())

# default
render_frames(m, 'default', frames=[64])

# arcade
m.loadScene('Arcade/Arcade.pyscene')
render_frames(m, 'arcade', frames=[64])

exit()
