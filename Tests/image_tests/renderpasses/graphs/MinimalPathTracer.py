from falcor import *

def render_graph_MinimalPathTracer():
    g = RenderGraph("MinimalPathTracer")
    loadRenderPassLibrary("AccumulatePass.dll")
    loadRenderPassLibrary("GBuffer.dll")
    loadRenderPassLibrary("MinimalPathTracer.dll")
    loadRenderPassLibrary("ToneMapper.dll")
    MinimalPathTracer = createPass("MinimalPathTracer")
    g.addPass(MinimalPathTracer, "MinimalPathTracer")
    GBufferRT = createPass("GBufferRT")
    g.addPass(GBufferRT, "GBufferRT")
    AccumulatePass = createPass("AccumulatePass")
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'operator': ToneMapOp.Linear, 'clamp': False, 'outputFormat': ResourceFormat.RGBA32Float})
    g.addPass(ToneMapper, "ToneMapper")
    g.addEdge("GBufferRT.posW", "MinimalPathTracer.posW")
    g.addEdge("GBufferRT.normW", "MinimalPathTracer.normalW")
    g.addEdge("GBufferRT.tangentW", "MinimalPathTracer.tangentW")
    g.addEdge("GBufferRT.faceNormalW", "MinimalPathTracer.faceNormalW")
    g.addEdge("GBufferRT.viewW", "MinimalPathTracer.viewW")
    g.addEdge("GBufferRT.specRough", "MinimalPathTracer.mtlSpecRough")
    g.addEdge("GBufferRT.emissive", "MinimalPathTracer.mtlEmissive")
    g.addEdge("GBufferRT.matlExtra", "MinimalPathTracer.mtlParams")
    g.addEdge("GBufferRT.diffuseOpacity", "MinimalPathTracer.mtlDiffOpacity")
    g.addEdge("MinimalPathTracer.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.markOutput("ToneMapper.dst")
    return g

MinimalPathTracer = render_graph_MinimalPathTracer()
try: m.addGraph(MinimalPathTracer)
except NameError: None
