def render_graph_MegakernelPathTracer():
    g = RenderGraph("MegakernelPathTracer")
    loadRenderPassLibrary("AccumulatePass.dll")
    loadRenderPassLibrary("GBuffer.dll")
    loadRenderPassLibrary("ToneMapper.dll")
    loadRenderPassLibrary("MegakernelPathTracer.dll")

    AccumulatePass = createPass("AccumulatePass", {'enabled': True})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMappingPass = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMappingPass, "ToneMappingPass")
    GBufferRT = createPass("GBufferRT", {'forceCullMode': False, 'cull': CullMode.CullBack, 'samplePattern': SamplePattern.Stratified, 'sampleCount': 16})
    g.addPass(GBufferRT, "GBufferRT")
    MegakernelPathTracer = createPass("MegakernelPathTracer", {'params': PathTracerParams(useVBuffer=0)})
    g.addPass(MegakernelPathTracer, "MegakernelPathTracer")

    g.addEdge("GBufferRT.posW", "MegakernelPathTracer.posW")
    g.addEdge("GBufferRT.normW", "MegakernelPathTracer.normalW")
    g.addEdge("GBufferRT.tangentW", "MegakernelPathTracer.tangentW")
    g.addEdge("GBufferRT.faceNormalW", "MegakernelPathTracer.faceNormalW")
    g.addEdge("GBufferRT.mtlData", "MegakernelPathTracer.mtlData")
    g.addEdge("GBufferRT.texC", "MegakernelPathTracer.texC")
    g.addEdge("GBufferRT.texGrads", "MegakernelPathTracer.texGrads")    # Required for texture filtering at primary hits (optional).
    g.addEdge("GBufferRT.viewW", "MegakernelPathTracer.viewW")          # Required for correct depth-of-field (optional).
    g.addEdge("GBufferRT.vbuffer", "MegakernelPathTracer.vbuffer")      # Required by ray footprint (optional).
    g.addEdge("MegakernelPathTracer.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMappingPass.src")

    g.markOutput("ToneMappingPass.dst")
    return g

MegakernelPathTracer = render_graph_MegakernelPathTracer()
try: m.addGraph(MegakernelPathTracer)
except NameError: None
