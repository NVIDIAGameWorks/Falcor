import sys
sys.path.append('..')
from helpers import render_frames
from graphs.SimplePostFX import SimplePostFX as g
from falcor import *

m.addGraph(g)

# default
render_frames(m, 'default')

# turn off features one by one
config = g.getPass('SimplePostFX').getDictionary()
config['bloomAmount']= 0.
g.updatePass('SimplePostFX', config)
render_frames(m, 'nobloom')

config['chromaticAberrationAmount'] = 0.0
g.updatePass('SimplePostFX', config)
render_frames(m, 'nochromatic')

config['barrelDistortAmount'] = 0.0
g.updatePass('SimplePostFX', config)
render_frames(m, 'nodistort')

config['enabled'] = False
g.updatePass('SimplePostFX', config)
render_frames(m, 'fullydisabled')

exit()
