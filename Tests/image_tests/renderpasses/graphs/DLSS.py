from falcor import *

def render_graph_DLSS():
    g = RenderGraph("DLSS")
    loadRenderPassLibrary("DLSSPass.dll")
    loadRenderPassLibrary("GBuffer.dll")
    GBufferRaster = createPass("GBufferRaster", {'samplePattern': SamplePattern.Halton})
    g.addPass(GBufferRaster, "GBufferRaster")
    DLSSPass = createPass("DLSSPass", {'motionVectorScale': DLSSMotionVectorScale.Relative})
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
