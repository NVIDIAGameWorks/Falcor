from falcor import *

def render_graph_CSM():
    g = RenderGraph("Cascaded Shadow Maps")
    g.addPass(createPass("DepthPass"), "DepthPrePass")
    g.addPass(createPass("CSM"), "ShadowPass")

    g.addEdge("DepthPrePass.depth", "ShadowPass.depth")

    g.markOutput("ShadowPass.visibility")

    return g

CSM = render_graph_CSM()
try: m.addGraph(CSM)
except NameError: None
