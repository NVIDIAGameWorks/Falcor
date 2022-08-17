from falcor import *

def render_graph_GBufferRT():
    loadRenderPassLibrary("GBuffer.dll")

    g = RenderGraph("GBufferRT")
    g.addPass(createPass("GBufferRT", {"useTraceRayInline": True}), "GBufferRT")

    g.markOutput("GBufferRT.posW")
    g.markOutput("GBufferRT.posW", TextureChannelFlags.Alpha)
    g.markOutput("GBufferRT.normW")
    g.markOutput("GBufferRT.tangentW")
    g.markOutput("GBufferRT.tangentW", TextureChannelFlags.Alpha)
    g.markOutput("GBufferRT.faceNormalW")
    g.markOutput("GBufferRT.texC")
    g.markOutput("GBufferRT.texGrads")
    g.markOutput("GBufferRT.texGrads", TextureChannelFlags.Alpha)
    g.markOutput("GBufferRT.mvec")
    g.markOutput("GBufferRT.mtlData")

    g.markOutput("GBufferRT.depth")
    g.markOutput("GBufferRT.vbuffer")
    g.markOutput("GBufferRT.vbuffer", TextureChannelFlags.Alpha)
    g.markOutput("GBufferRT.linearZ")
    g.markOutput("GBufferRT.mvecW")
    g.markOutput("GBufferRT.normWRoughnessMaterialID")
    g.markOutput("GBufferRT.normWRoughnessMaterialID", TextureChannelFlags.Alpha)
    g.markOutput("GBufferRT.diffuseOpacity")
    g.markOutput("GBufferRT.diffuseOpacity", TextureChannelFlags.Alpha)
    g.markOutput("GBufferRT.specRough")
    g.markOutput("GBufferRT.specRough", TextureChannelFlags.Alpha)
    g.markOutput("GBufferRT.emissive")
    g.markOutput("GBufferRT.viewW")
    g.markOutput("GBufferRT.disocclusion")

    return g

GBufferRT = render_graph_GBufferRT()
try: m.addGraph(GBufferRT)
except NameError: None
