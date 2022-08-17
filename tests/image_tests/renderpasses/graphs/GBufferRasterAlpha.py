from falcor import *

def render_graph_GBufferRaster():
    loadRenderPassLibrary("GBuffer.dll")

    g = RenderGraph("GBufferRaster")
    g.addPass(createPass("GBufferRaster"), "GBufferRaster")

    g.markOutput("GBufferRaster.posW")
    g.markOutput("GBufferRaster.posW", TextureChannelFlags.Alpha)

    return g

GBufferRaster = render_graph_GBufferRaster()
try: m.addGraph(GBufferRaster)
except NameError: None
