from falcor import *

def render_graph_RTXGI():
    g = RenderGraph("RTXGI")
    loadRenderPassLibrary("GBuffer.dll")
    loadRenderPassLibrary("RTXGIPass.dll")
    loadRenderPassLibrary("ToneMapper.dll")
    ToneMappingPass = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMappingPass, "ToneMappingPass")
    RTXGIPass = createPass("RTXGIPass", {'useVBuffer': False})
    g.addPass(RTXGIPass, "RTXGIPass")
    GBuffer = createPass("GBufferRaster")
    g.addPass(GBuffer, "GBuffer")
    g.addEdge("GBuffer.posW", "RTXGIPass.posW")
    g.addEdge("GBuffer.normW", "RTXGIPass.normalW")
    g.addEdge("GBuffer.tangentW", "RTXGIPass.tangentW")
    g.addEdge("GBuffer.faceNormalW", "RTXGIPass.faceNormalW")
    g.addEdge("GBuffer.texC", "RTXGIPass.texC")
    g.addEdge("GBuffer.texGrads", "RTXGIPass.texGrads") # This input is optional
    g.addEdge("GBuffer.mtlData", "RTXGIPass.mtlData")
    g.addEdge("GBuffer.depth", "RTXGIPass.depth") # This input is optional
    g.addEdge("RTXGIPass.output", "ToneMappingPass.src")
    g.markOutput("ToneMappingPass.dst")
    return g

RTXGI = render_graph_RTXGI()
try: m.addGraph(RTXGI)
except NameError: None
