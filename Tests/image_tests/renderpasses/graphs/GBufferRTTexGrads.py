from falcor import *

def render_graph_GBufferRT():
    loadRenderPassLibrary("GBuffer.dll")

    g = RenderGraph("GBufferRT")
    g.addPass(createPass("GBufferRT"), "GBufferRT")

    g.markOutput("GBufferRT.texGrads")
    g.markOutput("GBufferRT.texGrads", TextureChannelFlags.Alpha)

    return g

GBufferRT = render_graph_GBufferRT()
try: m.addGraph(GBufferRT)
except NameError: None
