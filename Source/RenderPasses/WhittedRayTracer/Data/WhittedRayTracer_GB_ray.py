from falcor import *

def render_graph_WhittedRayTracer():
    g = RenderGraph("DefaultRenderGraph")
    loadRenderPassLibrary("GBuffer.dll")
    loadRenderPassLibrary("WhittedRayTracer.dll")
    loadRenderPassLibrary("ToneMapper.dll")
    WhittedRayTracer = createPass("WhittedRayTracer", {'mUsingRasterizedGBuffer': False, 'mMaxBounces': 2, 'mComputeDirect': True, 'mTexLODMode': TextureLODMode.RayCones})
    g.addPass(WhittedRayTracer, "WhittedRayTracer")
    GBufferRT = createPass("GBufferRT", {'samplePattern': SamplePattern.Center, 'sampleCount': 1})
    g.addPass(GBufferRT, "GBufferRT")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureValue': 1.0, 'exposureCompensation': 1.0, 'operator': ToneMapOp.Linear})
    g.addPass(ToneMapper, "ToneMapper")
    g.addEdge("WhittedRayTracer.color", "ToneMapper.src")
    g.addEdge("GBufferRT.posW", "WhittedRayTracer.posW")
    g.addEdge("GBufferRT.normW", "WhittedRayTracer.normalW")
    g.addEdge("GBufferRT.tangentW", "WhittedRayTracer.tangentW")
    g.addEdge("GBufferRT.diffuseOpacity", "WhittedRayTracer.mtlDiffOpacity")
    g.addEdge("GBufferRT.specRough", "WhittedRayTracer.mtlSpecRough")
    g.addEdge("GBufferRT.matlExtra", "WhittedRayTracer.mtlParams")
    g.addEdge("GBufferRT.emissive", "WhittedRayTracer.mtlEmissive")
    g.addEdge("GBufferRT.faceNormalW", "WhittedRayTracer.faceNormalW")
    g.addEdge("GBufferRT.vbuffer", "WhittedRayTracer.vbuffer")
    g.markOutput("ToneMapper.dst")
    return g

WhittedRayTracerGraph = render_graph_WhittedRayTracer()
try: m.addGraph(WhittedRayTracerGraph)
except NameError: None
