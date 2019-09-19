# Scene
m.loadScene("Arcade/Arcade.fscene")

# Graphs
m.script("Data/ForwardRenderer.py")

# Window Configuration
m.resizeSwapChain(1920, 1080)
m.ui(true)

# Global Settings
t.now(0)
t.framerate(60)
# If framerate() is not zero, you can use the following function to set the start frame
# t.frame(0)

t.exitTime(600)

# Frame Capture
fc.outputDir("../../../Source/Mogwai")
fc.baseFilename("Mogwai")
g = m.activeGraph();
#fc.frames(g, [20, 50, 32])

# Video Capture
vc.outputDir(".")
vc.baseFilename("Mogwai")
vc.codec(Codec.H264)
vc.fps(60)
vc.bitrate(4.000000)
vc.gopSize(10)
#vc.ranges(g, [[30, 300]]);
