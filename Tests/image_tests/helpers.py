def render_frames(m, name, frames=[1], framerate=60, resolution=[1280,720]):
    m.resizeSwapChain(*resolution)
    m.ui = False
    m.clock.framerate = framerate
    m.clock.time = 0
    m.clock.pause()
    m.frameCapture.baseFilename = name

    frame = 0
    for capture_frame in frames:
        while frame < capture_frame:
            frame += 1
            m.clock.frame = frame
            m.renderFrame()
        m.frameCapture.capture()
