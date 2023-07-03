from falcor import *

def render_graph_TAA():
    testTAA = RenderGraph("TAA")
    GBufferRaster = createPass("GBufferRaster", {"samplePattern": 'Halton'})
    testTAA.addPass(GBufferRaster, "GBufferRaster")
    TAAPass = createPass("TAA")
    testTAA.addPass(TAAPass, "TAA")
    testTAA.addEdge("GBufferRaster.diffuseOpacity", "TAA.colorIn")
    testTAA.addEdge("GBufferRaster.mvec", "TAA.motionVecs")
    testTAA.markOutput("TAA.colorOut")
    return testTAA

TAA = render_graph_TAA()
try: m.addGraph(TAA)
except NameError: None
