from falcor import *

def render_graph_OptixDenoiser():
    g = RenderGraph("OptixDenoiser")
    loadRenderPassLibrary("AccumulatePass.dll")
    loadRenderPassLibrary("GBuffer.dll")
    loadRenderPassLibrary("OptixDenoiser.dll")
    loadRenderPassLibrary("PathTracer.dll")
    loadRenderPassLibrary("ToneMapper.dll")
    VBufferRT = createPass("VBufferRT")
    g.addPass(VBufferRT, "VBufferRT")
    AccumulatePass = createPass("AccumulatePass")
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMappingPass = createPass("ToneMapper")
    g.addPass(ToneMappingPass, "ToneMappingPass")
    PathTracer = createPass("PathTracer")
    g.addPass(PathTracer, "PathTracer")
    OptixDenoiser = createPass("OptixDenoiser")
    g.addPass(OptixDenoiser, "OptixDenoiser")
    g.addEdge("VBufferRT.vbuffer", "PathTracer.vbuffer")
    g.addEdge("PathTracer.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMappingPass.src")
    g.addEdge("ToneMappingPass.dst", "OptixDenoiser.color")
    g.addEdge("PathTracer.albedo", "OptixDenoiser.albedo")
    g.addEdge("PathTracer.normal", "OptixDenoiser.normal")
    g.addEdge("VBufferRT.mvec", "OptixDenoiser.mvec")

    # Color outputs
    g.markOutput("OptixDenoiser.output")
    g.markOutput("PathTracer.color")

    # OptixDenoiser inputs
    g.markOutput("ToneMappingPass.dst")
    g.markOutput("PathTracer.albedo")
    g.markOutput("PathTracer.normal")
    g.markOutput("VBufferRT.mvec")

    return g

OptixDenoiser = render_graph_OptixDenoiser()
try: m.addGraph(OptixDenoiser)
except NameError: None
