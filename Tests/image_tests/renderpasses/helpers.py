from pathlib import Path
from operator import itemgetter

def render_frames(ctx, name, frames=[1], framerate=60, resolution=[1280,720]):
    m, t, fc, renderFrame = itemgetter('m', 't', 'fc', 'renderFrame')(ctx)

    m.resizeSwapChain(*resolution)
    m.ui = False
    t.framerate = framerate
    t.time = 0
    t.pause()
    fc.baseFilename = name
    current_frame = 0
    for frame in frames:
        t.frame = frame
        while current_frame < frame:
            renderFrame()
            current_frame += 1
        fc.capture()
