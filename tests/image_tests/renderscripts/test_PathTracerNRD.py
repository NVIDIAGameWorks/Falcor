IMAGE_TEST = {
    'tolerance': 2e-6
}

# NOTE:
# NRD seems to be non-deterministic in some cases ("filteredSpecularRadianceHitDist" output).
# We're setting a larger threshold here to account for that.

import sys
sys.path.append('..')
from falcor import *
from helpers import render_frames

exec(open('../../../scripts/PathTracerNRD.py').read())

# default
render_frames(m, 'default', frames=[64])

# arcade
m.loadScene('Arcade/Arcade.pyscene')
render_frames(m, 'arcade', frames=[64])

exit()
