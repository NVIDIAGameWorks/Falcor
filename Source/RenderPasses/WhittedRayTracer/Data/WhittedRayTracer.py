from falcor import *

def render_graph_WhittedRayTracer():
    g = RenderGraph("WhittedRayTracer")
    loadRenderPassLibrary("GBuffer.dll")
    loadRenderPassLibrary("WhittedRayTracer.dll")
    loadRenderPassLibrary("ToneMapper.dll")
    WhittedRayTracer = createPass("WhittedRayTracer", {'maxBounces': 7, 'texLODMode': TexLODMode.RayCones, 'rayConeMode': RayConeMode.Unified, 'rayConeFilterMode': RayFootprintFilterMode.AnisotropicWhenRefraction, 'rayDiffFilterMode':RayFootprintFilterMode.AnisotropicWhenRefraction, 'useRoughnessToVariance': False})
    g.addPass(WhittedRayTracer, "WhittedRayTracer")
    GBufferRT = createPass("GBufferRT", {'samplePattern': SamplePattern.Center, 'sampleCount': 1})
    g.addPass(GBufferRT, "GBufferRT")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureValue': 1.0, 'exposureCompensation': 2.2, 'operator': ToneMapOp.Linear})
    g.addPass(ToneMapper, "ToneMapper")
    g.addEdge("WhittedRayTracer.color", "ToneMapper.src")
    g.addEdge("GBufferRT.posW", "WhittedRayTracer.posW")
    g.addEdge("GBufferRT.normW", "WhittedRayTracer.normalW")
    g.addEdge("GBufferRT.tangentW", "WhittedRayTracer.tangentW")
    g.addEdge("GBufferRT.faceNormalW", "WhittedRayTracer.faceNormalW")
    g.addEdge("GBufferRT.texC", "WhittedRayTracer.texC")
    g.addEdge("GBufferRT.texGrads", "WhittedRayTracer.texGrads")
    g.addEdge("GBufferRT.mtlData", "WhittedRayTracer.mtlData")
    g.addEdge("GBufferRT.vbuffer", "WhittedRayTracer.vbuffer")
    g.markOutput("ToneMapper.dst")
    return g

WhittedRayTracerGraph = render_graph_WhittedRayTracer()
try: m.addGraph(WhittedRayTracerGraph)
except NameError: None
