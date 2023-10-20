import falcor

def render_frames(m, name, frames=[1], framerate=60, resolution=[640,360]):
    m.resizeFrameBuffer(*resolution)
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
        if "IMAGE_TEST_RUN_ONLY" in falcor.__dict__:
            continue
        m.frameCapture.capture()
