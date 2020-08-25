from falcor import *

def render_graph_MinimalPathTracer():
    g = RenderGraph("MinimalPathTracer")
    loadRenderPassLibrary("AccumulatePass.dll")
    loadRenderPassLibrary("GBuffer.dll")
    loadRenderPassLibrary("MinimalPathTracer.dll")
    loadRenderPassLibrary("ToneMapper.dll")
    AccumulatePass = createPass("AccumulatePass", {'enableAccumulation': True, 'precisionMode': AccumulatePrecision.Single})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureValue': 0.0})
    g.addPass(ToneMapper, "ToneMapper")
    MinimalPathTracer = createPass("MinimalPathTracer", {'mMaxBounces': 3, 'mComputeDirect': True})
    g.addPass(MinimalPathTracer, "MinimalPathTracer")
    GBufferRT = createPass("GBufferRT", {'samplePattern': SamplePattern.Stratified, 'sampleCount': 16})
    g.addPass(GBufferRT, "GBufferRT")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.addEdge("GBufferRT.posW", "MinimalPathTracer.posW")
    g.addEdge("GBufferRT.normW", "MinimalPathTracer.normalW")
    g.addEdge("GBufferRT.tangentW", "MinimalPathTracer.tangentW")
    g.addEdge("GBufferRT.faceNormalW", "MinimalPathTracer.faceNormalW")
    g.addEdge("GBufferRT.viewW", "MinimalPathTracer.viewW")
    g.addEdge("GBufferRT.diffuseOpacity", "MinimalPathTracer.mtlDiffOpacity")
    g.addEdge("GBufferRT.specRough", "MinimalPathTracer.mtlSpecRough")
    g.addEdge("GBufferRT.emissive", "MinimalPathTracer.mtlEmissive")
    g.addEdge("GBufferRT.matlExtra", "MinimalPathTracer.mtlParams")
    g.addEdge("MinimalPathTracer.color", "AccumulatePass.input")
    g.markOutput("ToneMapper.dst")
    return g

MinimalPathTracer = render_graph_MinimalPathTracer()
try: m.addGraph(MinimalPathTracer)
except NameError: None
