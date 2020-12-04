def render_frames(m, name, frames=[1], framerate=60, resolution=[1280,720]):
    m.resizeSwapChain(*resolution)
    m.ui = False
    m.clock.framerate = framerate
    m.clock.time = 0
    m.clock.pause()
    m.frameCapture.baseFilename = name
    current_frame = 0
    for frame in frames:
        m.clock.frame = frame
        while current_frame < frame:
            m.renderFrame()
            current_frame += 1
        m.frameCapture.capture()
