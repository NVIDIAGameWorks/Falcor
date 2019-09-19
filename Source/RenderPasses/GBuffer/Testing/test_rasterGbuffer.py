def render_graph_GBufferRaster():
    loadRenderPassLibrary("GBuffer.dll")
    
    tracer = RenderGraph("RasterGBuffer")
    tracer.addPass(RenderPass("GBufferRaster"), "GBufferRaster")

    tracer.markOutput("GBufferRaster.posW")
    tracer.markOutput("GBufferRaster.normW")
    tracer.markOutput("GBufferRaster.bitangentW")
    tracer.markOutput("GBufferRaster.texC")
    tracer.markOutput("GBufferRaster.diffuseOpacity")
    tracer.markOutput("GBufferRaster.specRough")
    tracer.markOutput("GBufferRaster.emissive")
    tracer.markOutput("GBufferRaster.matlExtra")

    return tracer

GBufferRaster = render_graph_GBufferRaster()
try: m.addGraph(GBufferRaster)
except NameError: None
