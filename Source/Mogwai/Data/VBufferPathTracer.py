def render_graph_VBufferPathTracerGraph():
    g = RenderGraph("VBufferPathTracerGraph")
    loadRenderPassLibrary("AccumulatePass.dll")
    loadRenderPassLibrary("GBuffer.dll")
    loadRenderPassLibrary("ToneMapper.dll")
    loadRenderPassLibrary("MegakernelPathTracer.dll")
    AccumulatePass = RenderPass("AccumulatePass", {'enableAccumulation': True})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMappingPass = RenderPass("ToneMapper", {'autoExposure': False, 'exposureValue': 0.0})
    g.addPass(ToneMappingPass, "ToneMappingPass")
    VBufferRT = RenderPass("VBufferRT", {'samplePattern': SamplePattern.Stratified, 'sampleCount': 16})
    g.addPass(VBufferRT, "VBufferRT")
    MegakernelPathTracer = RenderPass("MegakernelPathTracer", {'mSharedParams': PathTracerParams(useVBuffer=1, , useAnalyticLights=0)})
    g.addPass(MegakernelPathTracer, "MegakernelPathTracer")
    g.addEdge("VBufferRT.vbuffer", "MegakernelPathTracer.vbuffer")
    g.addEdge("MegakernelPathTracer.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMappingPass.src")
    g.markOutput("ToneMappingPass.dst")
    return g

VBufferPathTracerGraph = render_graph_VBufferPathTracerGraph()
try: m.addGraph(VBufferPathTracerGraph)
except NameError: None
