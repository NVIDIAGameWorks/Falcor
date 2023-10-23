import sys
sys.path.append('..')
from falcor import *
from helpers import render_frames

exec(open('../../../scripts/WARDiffPathTracer.py').read())

# default
render_frames(m, 'default', frames=[64])

exit()
