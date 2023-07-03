from falcor import *

def render_graph_MVecRaster():
    g = RenderGraph("MVecRaster")
    g.addPass(createPass("GBufferRaster", {'samplePattern': 'Stratified', 'sampleCount': 16}), "GBufferRaster")

    g.markOutput("GBufferRaster.mvec")

    return g

MVecRaster = render_graph_MVecRaster()
try: m.addGraph(MVecRaster)
except NameError: None
