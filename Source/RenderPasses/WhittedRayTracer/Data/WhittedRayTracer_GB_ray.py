from falcor import *

def render_graph_WhittedRayTracer():
    g = RenderGraph("DefaultRenderGraph")
    loadRenderPassLibrary("GBuffer.dll")
    loadRenderPassLibrary("WhittedRatTracer.dll")
    loadRenderPassLibrary("ToneMapper.dll")
    WhittedRayTracer = RenderPass("WhittedRayTracer", {'mUsingRasterizedGBuffer': False, 'mMaxBounces': 2, 'mComputeDirect': True, 'mUseAnalyticLights': 1, 'mUseEmissiveLights': 1, 'mUseEnvLight': 0, 'mUseEnvBackground': 1})
    g.addPass(WhittedRayTracer, "WhittedRayTracer")
    GBufferRT = RenderPass("GBufferRT", {'samplePattern': SamplePattern.Center, 'sampleCount': 16, 'forceCullMode': False, 'cull': CullMode.CullBack, 'useBentShadingNormals': True, 'texLOD': LODMode.UseMip0})
    g.addPass(GBufferRT, "GBufferRT")
    ToneMapper = RenderPass("ToneMapper", {'exposureCompensation': 0.0, 'autoExposure': True, 'exposureValue': 0.0, 'filmSpeed': 100.0, 'whiteBalance': False, 'whitePoint': 6500.0, 'operator': ToneMapOp.Aces, 'clamp': True, 'whiteMaxLuminance': 1.0, 'whiteScale': 11.199999809265137})
    g.addPass(ToneMapper, "ToneMapper")
    g.addEdge("WhittedRayTracer.color", "ToneMapper.src")
    g.addEdge("GBufferRT.posW", "WhittedRayTracer.posW")
    g.addEdge("GBufferRT.normW", "WhittedRayTracer.normalW")
    g.addEdge("GBufferRT.bitangentW", "WhittedRayTracer.bitangentW")
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
