from falcor import *

def render_graph_WhittedRayTracer():
    g = RenderGraph("DefaultRenderGraph")
    loadRenderPassLibrary("GBuffer.dll")
    loadRenderPassLibrary("WhittedRatTracer.dll")
    loadRenderPassLibrary("ToneMapper.dll")
    WhittedRayTracer = RenderPass("WhittedRayTracer", {'mUsingRasterizedGBuffer': True, 'mMaxBounces': 1, 'mComputeDirect': True, 'mUseAnalyticLights': 1, 'mUseEmissiveLights': 1, 'mUseEnvLight': 0, 'mUseEnvBackground': 1})
    g.addPass(WhittedRayTracer, "WhittedRayTracer")
    GBufferRaster = RenderPass("GBufferRaster", {'samplePattern': SamplePattern.Center, 'sampleCount': 16, 'forceCullMode': False, 'cull': CullMode.CullBack, 'useBentShadingNormals': True, 'texLOD': LODMode.UseMip0})
    g.addPass(GBufferRaster, "GBufferRaster")
    ToneMapper = RenderPass("ToneMapper", {'exposureCompensation': 0.0, 'autoExposure': True, 'exposureValue': 0.0, 'filmSpeed': 100.0, 'whiteBalance': False, 'whitePoint': 6500.0, 'operator': ToneMapOp.Aces, 'clamp': True, 'whiteMaxLuminance': 1.0, 'whiteScale': 11.199999809265137})
    g.addPass(ToneMapper, "ToneMapper")
    g.addEdge("WhittedRayTracer.color", "ToneMapper.src")
    g.addEdge("GBufferRaster.posW", "WhittedRayTracer.posW")
    g.addEdge("GBufferRaster.normW", "WhittedRayTracer.normalW")
    g.addEdge("GBufferRaster.bitangentW", "WhittedRayTracer.bitangentW")
    g.addEdge("GBufferRaster.diffuseOpacity", "WhittedRayTracer.mtlDiffOpacity")
    g.addEdge("GBufferRaster.specRough", "WhittedRayTracer.mtlSpecRough")
    g.addEdge("GBufferRaster.matlExtra", "WhittedRayTracer.mtlParams")
    g.addEdge("GBufferRaster.emissive", "WhittedRayTracer.mtlEmissive")
    g.addEdge("GBufferRaster.faceNormalW", "WhittedRayTracer.faceNormalW")
    g.addEdge("GBufferRaster.surfSpreadAngle", "WhittedRayTracer.surfSpreadAngle")
    g.addEdge("GBufferRaster.rayDifferentialX", "WhittedRayTracer.rayDifferentialX")
    g.addEdge("GBufferRaster.rayDifferentialY", "WhittedRayTracer.rayDifferentialY")
    g.addEdge("GBufferRaster.rayDifferentialZ", "WhittedRayTracer.rayDifferentialZ")
    g.markOutput("ToneMapper.dst")
    return g

WhittedRayTracerGraph = render_graph_WhittedRayTracer()
try: m.addGraph(WhittedRayTracerGraph)
except NameError: None
