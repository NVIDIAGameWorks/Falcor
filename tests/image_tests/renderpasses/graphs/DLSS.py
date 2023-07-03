from falcor import *

def render_graph_DLSS():
    g = RenderGraph("DLSS")
    GBufferRaster = createPass("GBufferRaster", {'samplePattern': 'Halton'})
    g.addPass(GBufferRaster, "GBufferRaster")
    DLSSPass = createPass("DLSSPass", {'motionVectorScale': 'Relative'})
    g.addPass(DLSSPass, "DLSSPass")
    g.addEdge("GBufferRaster.mvec", "DLSSPass.mvec")
    g.addEdge("GBufferRaster.depth", "DLSSPass.depth")
    g.addEdge("GBufferRaster.diffuseOpacity", "DLSSPass.color")
    g.markOutput("GBufferRaster.mvec")
    g.markOutput("GBufferRaster.depth")
    g.markOutput("GBufferRaster.diffuseOpacity")
    g.markOutput("DLSSPass.output")
    return g

DLSS = render_graph_DLSS()
try: m.addGraph(DLSS)
except NameError: None
