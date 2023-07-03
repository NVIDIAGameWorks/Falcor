from falcor import *

def render_graph_MVecRT():
    g = RenderGraph("MVecRT")
    g.addPass(createPass("GBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 16}), "GBufferRT")

    g.markOutput("GBufferRT.mvec")

    return g

MVecRT = render_graph_MVecRT()
try: m.addGraph(MVecRT)
except NameError: None
