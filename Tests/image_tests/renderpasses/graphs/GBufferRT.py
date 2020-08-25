from falcor import *

def render_graph_GBufferRT():
    loadRenderPassLibrary("GBuffer.dll")

    g = RenderGraph("GBufferRT")
    g.addPass(createPass("GBufferRT"), "GBufferRT")

    g.markOutput("GBufferRT.posW")
    g.markOutput("GBufferRT.normW")
    g.markOutput("GBufferRT.tangentW")
    g.markOutput("GBufferRT.texC")
    g.markOutput("GBufferRT.diffuseOpacity")
    g.markOutput("GBufferRT.specRough")
    g.markOutput("GBufferRT.emissive")
    g.markOutput("GBufferRT.matlExtra")

    g.markOutput("GBufferRT.vbuffer")
    g.markOutput("GBufferRT.mvec")
    g.markOutput("GBufferRT.faceNormalW")
    g.markOutput("GBufferRT.viewW")

    return g

GBufferRT = render_graph_GBufferRT()
try: m.addGraph(GBufferRT)
except NameError: None
