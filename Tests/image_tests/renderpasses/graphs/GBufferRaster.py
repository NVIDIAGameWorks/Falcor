from falcor import *

def render_graph_GBufferRaster():
    loadRenderPassLibrary("GBuffer.dll")

    g = RenderGraph("GBufferRaster")
    g.addPass(createPass("GBufferRaster"), "GBufferRaster")

    g.markOutput("GBufferRaster.posW")
    g.markOutput("GBufferRaster.normW")
    g.markOutput("GBufferRaster.tangentW")
    g.markOutput("GBufferRaster.texC")
    g.markOutput("GBufferRaster.diffuseOpacity")
    g.markOutput("GBufferRaster.specRough")
    g.markOutput("GBufferRaster.emissive")
    g.markOutput("GBufferRaster.matlExtra")

    g.markOutput("GBufferRaster.vbuffer")
    g.markOutput("GBufferRaster.mvec")
    g.markOutput("GBufferRaster.faceNormalW")
    g.markOutput("GBufferRaster.pnFwidth")
    g.markOutput("GBufferRaster.linearZ")
    g.markOutput("GBufferRaster.depth")

    return g

GBufferRaster = render_graph_GBufferRaster()
try: m.addGraph(GBufferRaster)
except NameError: None
