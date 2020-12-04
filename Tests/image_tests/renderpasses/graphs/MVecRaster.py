from falcor import *

def render_graph_MVecRaster():
    loadRenderPassLibrary("GBuffer.dll")

    g = RenderGraph("MVecRaster")
    g.addPass(createPass("GBufferRaster", {'samplePattern': SamplePattern.Stratified, 'sampleCount': 16}), "GBufferRaster")

    g.markOutput("GBufferRaster.mvec")

    return g

MVecRaster = render_graph_MVecRaster()
try: m.addGraph(MVecRaster)
except NameError: None
