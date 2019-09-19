def render_graph_GBufferRT():
    loadRenderPassLibrary("GBuffer.dll")
    
    tracer = RenderGraph("RtGbuffer")
    tracer.addPass(RenderPass("GBufferRT"), "GBufferRT")

    tracer.markOutput("GBufferRT.posW")
    tracer.markOutput("GBufferRT.normW")
    tracer.markOutput("GBufferRT.bitangentW")
    tracer.markOutput("GBufferRT.texC")
    tracer.markOutput("GBufferRT.diffuseOpacity")
    tracer.markOutput("GBufferRT.specRough")
    tracer.markOutput("GBufferRT.emissive")
    tracer.markOutput("GBufferRT.matlExtra")

    return tracer

GBufferRT = render_graph_GBufferRT()
try: m.addGraph(GBufferRT)
except NameError: None