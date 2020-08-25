from falcor import *

def render_graph_VBufferRT():
    loadRenderPassLibrary("GBuffer.dll")

    g = RenderGraph("VBufferRT")
    g.addPass(createPass("VBufferRT"), "VBufferRT")

    g.markOutput("VBufferRT.vbuffer")

    return g

VBufferRT = render_graph_VBufferRT()
try: m.addGraph(VBufferRT)
except NameError: None
