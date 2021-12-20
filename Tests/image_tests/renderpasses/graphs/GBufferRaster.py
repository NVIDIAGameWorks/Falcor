from falcor import *

def render_graph_GBufferRaster():
    loadRenderPassLibrary("GBuffer.dll")

    g = RenderGraph("GBufferRaster")
    g.addPass(createPass("GBufferRaster"), "GBufferRaster")

    g.markOutput("GBufferRaster.posW")
    g.markOutput("GBufferRaster.posW", TextureChannelFlags.Alpha)
    g.markOutput("GBufferRaster.normW")
    g.markOutput("GBufferRaster.tangentW")
    g.markOutput("GBufferRaster.tangentW", TextureChannelFlags.Alpha)
    g.markOutput("GBufferRaster.faceNormalW")
    g.markOutput("GBufferRaster.texC")
    g.markOutput("GBufferRaster.texGrads")
    g.markOutput("GBufferRaster.texGrads", TextureChannelFlags.Alpha)
    g.markOutput("GBufferRaster.mvec")
    g.markOutput("GBufferRaster.mtlData")

    g.markOutput("GBufferRaster.vbuffer")
    g.markOutput("GBufferRaster.vbuffer", TextureChannelFlags.Alpha)
    g.markOutput("GBufferRaster.depth")
    g.markOutput("GBufferRaster.diffuseOpacity")
    g.markOutput("GBufferRaster.diffuseOpacity", TextureChannelFlags.Alpha)
    g.markOutput("GBufferRaster.specRough")
    g.markOutput("GBufferRaster.specRough", TextureChannelFlags.Alpha)
    g.markOutput("GBufferRaster.emissive")
    g.markOutput("GBufferRaster.viewW")
    g.markOutput("GBufferRaster.pnFwidth")
    g.markOutput("GBufferRaster.linearZ")

    return g

GBufferRaster = render_graph_GBufferRaster()
try: m.addGraph(GBufferRaster)
except NameError: None
