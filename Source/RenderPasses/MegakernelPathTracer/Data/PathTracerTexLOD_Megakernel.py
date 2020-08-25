def render_graph_PathTracerGraph():
    g = RenderGraph("PathTracerGraph")
    loadRenderPassLibrary("AccumulatePass.dll")
    loadRenderPassLibrary("GBuffer.dll")
    loadRenderPassLibrary("ToneMapper.dll")
    loadRenderPassLibrary("MegakernelPathTracer.dll")
    AccumulatePass = createPass("AccumulatePass", {'enableAccumulation': True})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMappingPass = createPass("ToneMapper", {'autoExposure': False, 'exposureValue': 0.0})
    g.addPass(ToneMappingPass, "ToneMappingPass")
    GBufferRT = createPass("GBufferRT", {'forceCullMode': False, 'cull': CullMode.CullBack, 'samplePattern': SamplePattern.Stratified, 'sampleCount': 16})
    GBufferRaster = createPass("GBufferRaster", {'forceCullMode': False, 'cull': CullMode.CullBack, 'samplePattern': SamplePattern.Stratified, 'sampleCount': 16})      # viewW not exported ? Not compatible with Path Tracers anymore ?
    g.addPass(GBufferRT, "GBuffer")
#    MegakernelPathTracer = createPass("MegakernelPathTracer", {'mSharedParams': PathTracerParams(useVBuffer=0, rayFootprintMode=0)})  # Generates an error apparently because of rayFootprintMode being unsigned, is there a specific syntac to use ?
    MegakernelPathTracer = createPass("MegakernelPathTracer", {'mSharedParams': PathTracerParams(useVBuffer=0)})
    g.addPass(MegakernelPathTracer, "PathTracer")
    g.addEdge("GBuffer.vbuffer", "PathTracer.vbuffer")      # Required by Ray Footprint.
    g.addEdge("GBuffer.posW", "PathTracer.posW")
    g.addEdge("GBuffer.normW", "PathTracer.normalW")
    g.addEdge("GBuffer.tangentW", "PathTracer.tangentW")
    g.addEdge("GBuffer.faceNormalW", "PathTracer.faceNormalW")
    g.addEdge("GBuffer.viewW", "PathTracer.viewW")
    g.addEdge("GBuffer.diffuseOpacity", "PathTracer.mtlDiffOpacity")
    g.addEdge("GBuffer.specRough", "PathTracer.mtlSpecRough")
    g.addEdge("GBuffer.emissive", "PathTracer.mtlEmissive")
    g.addEdge("GBuffer.matlExtra", "PathTracer.mtlParams")
    g.addEdge("PathTracer.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMappingPass.src")
    g.markOutput("ToneMappingPass.dst")
    return g

PathTracerGraph = render_graph_PathTracerGraph()
try: m.addGraph(PathTracerGraph)
except NameError: None
