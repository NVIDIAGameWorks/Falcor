from falcor import *

def render_graph_RTXDI():
    g = RenderGraph("RTXDI")
    loadRenderPassLibrary("AccumulatePass.dll")
    loadRenderPassLibrary("GBuffer.dll")
    loadRenderPassLibrary("RTXDIPass.dll")
    loadRenderPassLibrary("ToneMapper.dll")
    VBufferRT = createPass("VBufferRT")
    g.addPass(VBufferRT, "VBufferRT")
    RTXDIPass = createPass("RTXDIPass")
    g.addPass(RTXDIPass, "RTXDIPass")
    AccumulatePass = createPass("AccumulatePass", {'enabled': False, 'precisionMode': AccumulatePrecision.Single})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")
    g.addEdge("VBufferRT.vbuffer", "RTXDIPass.vbuffer")
    g.addEdge("VBufferRT.mvec", "RTXDIPass.mvec")
    g.addEdge("RTXDIPass.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.markOutput("ToneMapper.dst")
    return g

RTXDI = render_graph_RTXDI()
try: m.addGraph(RTXDI)
except NameError: None
