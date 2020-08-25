from falcor import *

def render_graph_WhittedRayTracer():
    g = RenderGraph("DefaultRenderGraph")
    loadRenderPassLibrary("GBuffer.dll")
    loadRenderPassLibrary("WhittedRayTracer.dll")
    loadRenderPassLibrary("ToneMapper.dll")
    WhittedRayTracer = createPass("WhittedRayTracer", {'mUsingRasterizedGBuffer': True, 'mMaxBounces': 2, 'mComputeDirect': True, 'mTexLODMode': TextureLODMode.RayCones})
    g.addPass(WhittedRayTracer, "WhittedRayTracer")
    GBufferRaster = createPass("GBufferRaster", {'samplePattern': SamplePattern.Center, 'sampleCount': 1, 'forceCullMode': True, 'cull': CullMode.CullNone,})
    g.addPass(GBufferRaster, "GBufferRaster")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureValue': 1.0, 'exposureCompensation': 1.0, 'operator': ToneMapOp.Linear})
    g.addPass(ToneMapper, "ToneMapper")
    g.addEdge("WhittedRayTracer.color", "ToneMapper.src")
    g.addEdge("GBufferRaster.posW", "WhittedRayTracer.posW")
    g.addEdge("GBufferRaster.normW", "WhittedRayTracer.normalW")
    g.addEdge("GBufferRaster.tangentW", "WhittedRayTracer.tangentW")
    g.addEdge("GBufferRaster.diffuseOpacity", "WhittedRayTracer.mtlDiffOpacity")
    g.addEdge("GBufferRaster.specRough", "WhittedRayTracer.mtlSpecRough")
    g.addEdge("GBufferRaster.matlExtra", "WhittedRayTracer.mtlParams")
    g.addEdge("GBufferRaster.emissive", "WhittedRayTracer.mtlEmissive")
    g.addEdge("GBufferRaster.faceNormalW", "WhittedRayTracer.faceNormalW")
    g.addEdge("GBufferRaster.vbuffer", "WhittedRayTracer.vbuffer")
    g.addEdge("GBufferRaster.surfSpreadAngle", "WhittedRayTracer.surfSpreadAngle")  # you can get rid of this if you do NOT want to use the ray cones method from Ray Tracing Gems 1 (chapter 20)
    g.markOutput("ToneMapper.dst")
    return g

WhittedRayTracerGraph = render_graph_WhittedRayTracer()
try: m.addGraph(WhittedRayTracerGraph)
except NameError: None
