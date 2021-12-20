IMAGE_TEST = {
    'skipped': 'Skipped due to instability on testing agents.'
}

import sys
sys.path.append('..')
from helpers import render_frames
from graphs.OptixDenoiser import OptixDenoiser as g
from falcor import *

m.addGraph(g)
m.loadScene('Arcade/Arcade.pyscene')

# default
render_frames(m, 'default', frames=[64])

exit()
