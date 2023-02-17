from falcor import *

def render_graph_VBufferRaster():
    g = RenderGraph("VBufferRaster")
    g.addPass(createPass("VBufferRaster"), "VBufferRaster")

    g.markOutput("VBufferRaster.vbuffer")
    g.markOutput("VBufferRaster.vbuffer", TextureChannelFlags.Alpha)
    g.markOutput("VBufferRaster.depth")
    g.markOutput("VBufferRaster.mvec")
    g.markOutput("VBufferRaster.mask")

    return g

VBufferRaster = render_graph_VBufferRaster()
try: m.addGraph(VBufferRaster)
except NameError: None
