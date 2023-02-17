from falcor import *

def render_graph_RTXDI():
    g = RenderGraph("RTXDI")
    ToneMappingPass = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMappingPass, "ToneMappingPass")
    RTXDIPass = createPass("RTXDIPass", {'useVBuffer': False})
    g.addPass(RTXDIPass, "RTXDIPass")
    VBuffer = createPass("VBufferRT")
    g.addPass(VBuffer, "VBuffer")
    g.addEdge("VBuffer.vbuffer", "RTXDIPass.vbuffer")
    g.addEdge("VBuffer.mvec", "RTXDIPass.mvec")
    g.addEdge("RTXDIPass.color", "ToneMappingPass.src")
    g.markOutput("ToneMappingPass.dst")
    return g

RTXDI = render_graph_RTXDI()
try: m.addGraph(RTXDI)
except NameError: None
