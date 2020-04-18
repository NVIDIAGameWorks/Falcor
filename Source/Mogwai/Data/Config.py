# Scene
m.loadScene("Arcade/Arcade.fscene")

# Graphs
m.script("Data/ForwardRenderer.py")

# Window Configuration
m.resizeSwapChain(1920, 1080)
m.ui = True

# Global Settings
t.time = 0
t.framerate = 60
# If framerate is not zero, you can use the frame property to set the start frame
# t.frame = 0

# Frame Capture
fc.outputDir = "../../../Source/Mogwai"
fc.baseFilename = "Mogwai"
#fc.addFrames(m.activeGraph, [20, 50, 32])

# Video Capture
vc.outputDir = "."
vc.baseFilename = "Mogwai"
vc.codec = Codec.H264
vc.fps = 60
vc.bitrate = 4.0
vc.gopSize = 10
#vc.addRanges(m.activeGraph, [[30, 300]])
